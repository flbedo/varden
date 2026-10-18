"""Hybrid search over the local knowledge base.

The original Hyperion search mixed BM25 and normalized sentence embeddings. This
module keeps that useful part, fixes score-to-document alignment, and removes the
old chat history and cloud-LLM code from the retrieval path.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize user and document text without changing its meaning."""

    return " ".join(unicodedata.normalize("NFKC", text or "").split()).strip()


def _tokens(text: str) -> list[str]:
    return [token.strip("-_") for token in TOKEN_RE.findall(text.lower()) if token.strip("-_")]


class _BM25:
    """Small BM25 implementation; avoids NLTK and rank-bm25 runtime downloads."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.lengths = np.asarray([len(document) for document in documents], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if documents else 0.0
        self.term_counts = [Counter(document) for document in documents]

        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        count = len(documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        result = np.zeros(len(self.documents), dtype=np.float32)
        if not query_tokens or not self.documents or not self.avg_length:
            return result

        for index, frequencies in enumerate(self.term_counts):
            length_factor = self.k1 * (
                1.0 - self.b + self.b * float(self.lengths[index]) / self.avg_length
            )
            score = 0.0
            for term in set(query_tokens):
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * (
                        frequency * (self.k1 + 1.0) / (frequency + length_factor)
                    )
            result[index] = score
        return result


def _document_parts(path: Path) -> Iterable[tuple[str, int | None]]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        for page_number, page in enumerate(PdfReader(path).pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                yield text, page_number
        return

    text = normalize_text(path.read_text(encoding="utf-8"))
    if text:
        yield text, None


def _chunk_text(text: str, size: int = 180, overlap: int = 30) -> Iterable[str]:
    words = text.split()
    if not words:
        return

    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start : start + size]
        if chunk:
            yield " ".join(chunk)
        if start + size >= len(words):
            break


def _iter_files(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Неподдерживаемый формат базы знаний: {source.suffix}")
        return [source]
    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    raise FileNotFoundError(f"База знаний не найдена: {source}")


def load_chunks(source: str | Path) -> list[dict[str, Any]]:
    """Read TXT, Markdown, and PDF files and retain source metadata."""

    source_path = Path(source).expanduser().resolve()
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()

    for path in _iter_files(source_path):
        filename = str(path.relative_to(source_path)) if source_path.is_dir() else path.name
        chunk_number = 0
        for document_text, page in _document_parts(path):
            for text in _chunk_text(document_text):
                duplicate_key = text.casefold()
                if duplicate_key in seen:
                    continue
                seen.add(duplicate_key)
                chunk_number += 1
                page_marker = f"p{page}" if page is not None else "text"
                chunks.append(
                    {
                        "text": text,
                        "source": filename,
                        "page": page,
                        "chunk_id": f"{filename}:{page_marker}:{chunk_number}",
                    }
                )

    if not chunks:
        raise ValueError(f"В базе знаний нет непустых поддерживаемых документов: {source_path}")
    return chunks


class TextSearchEngine:
    """Hybrid BM25 + cosine search, based on the original Hyperion retriever."""

    def __init__(
        self,
        file_path: str | Path = "faq.txt",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunks: list[str] | list[dict[str, Any]] | None = None,
        embedding_model: Any | None = None,
    ):
        if chunks is None:
            self.chunks = load_chunks(file_path)
        else:
            self.chunks = []
            for index, chunk in enumerate(chunks, start=1):
                data = dict(chunk) if isinstance(chunk, dict) else {"text": chunk}
                text = normalize_text(str(data.get("text", "")))
                if not text:
                    continue
                data.update(
                    text=text,
                    source=data.get("source", Path(file_path).name),
                    page=data.get("page"),
                    chunk_id=data.get("chunk_id", f"{Path(file_path).name}:text:{index}"),
                )
                self.chunks.append(data)
        if not self.chunks:
            raise ValueError("Нельзя построить индекс без текстовых фрагментов")

        if embedding_model is None:
            embedding_model = SentenceTransformer(model_name, local_files_only=True)

        self.model = embedding_model
        self.tokenized_chunks = [_tokens(chunk["text"]) for chunk in self.chunks]
        self.bm25 = _BM25(self.tokenized_chunks)
        embeddings = self.model.encode(
            [chunk["text"] for chunk in self.chunks],
            show_progress_bar=False,
            batch_size=32,
        )
        self.embeddings = self._normalize_embeddings(embeddings)

    @staticmethod
    def _normalize_embeddings(embeddings: Any) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.25,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return best chunks; ``alpha`` is the lexical-search weight."""

        query = normalize_text(query)
        if not query or k <= 0:
            return []
        alpha = min(1.0, max(0.0, alpha))

        query_embedding = self.model.encode([query], show_progress_bar=False)
        normalized_query = self._normalize_embeddings(query_embedding)[0]
        similarities = self.embeddings @ normalized_query

        bm25_scores = self.bm25.scores(_tokens(query))
        max_bm25 = float(bm25_scores.max()) if len(bm25_scores) else 0.0
        normalized_bm25 = bm25_scores / max_bm25 if max_bm25 > 0 else bm25_scores

        # Unlike the old implementation, both arrays are indexed by document ID.
        dense_scores = np.clip(similarities, 0.0, 1.0)
        combined_scores = alpha * normalized_bm25 + (1.0 - alpha) * dense_scores
        if threshold is None:
            eligible_indices = np.arange(len(self.chunks))
        else:
            eligible_indices = np.flatnonzero(similarities >= threshold)
        candidate_indices = eligible_indices[
            np.argsort(combined_scores[eligible_indices])[::-1]
        ]

        results: list[dict[str, Any]] = []
        selected_token_sets: list[set[str]] = []
        for raw_index in candidate_indices:
            index = int(raw_index)
            similarity = float(similarities[index])
            token_set = set(self.tokenized_chunks[index])
            if any(
                token_set
                and previous
                and len(token_set & previous) / min(len(token_set), len(previous)) >= 0.9
                for previous in selected_token_sets
            ):
                continue

            chunk = self.chunks[index]
            results.append(
                {
                    "text": chunk["text"],
                    "score": float(combined_scores[index]),
                    "similarity": similarity,
                    "bm25_score": float(bm25_scores[index]),
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "chunk_id": chunk.get("chunk_id", str(index)),
                }
            )
            selected_token_sets.append(token_set)
            if len(results) == k:
                break

        return results


@lru_cache(maxsize=1)
def get_search_engine() -> TextSearchEngine:
    """Load the embedding model and build the static index once per process."""
    import config

    return TextSearchEngine(
        file_path=config.KNOWLEDGE_BASE_PATH,
        model_name=config.RAG_EMBEDDING_MODEL,
    )


def search(question: str) -> dict[str, Any]:
    """Search the knowledge base and return context, sources, and confidence."""
    import config

    question = normalize_text(question)
    if not question:
        return {"context": "", "sources": [], "confidence": 0.0, "search_time": 0.0}

    started = time.perf_counter()
    matches = get_search_engine().search(
        question,
        k=config.RAG_TOP_K,
        threshold=config.RAG_THRESHOLD,
    )

    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    for match in matches:
        location = f", страница {match['page']}" if match["page"] is not None else ""
        context_parts.append(
            f"[Источник: {match['source']}{location}; фрагмент: {match['chunk_id']}]\n"
            f"{match['text']}"
        )
        sources.append(
            {
                "id": match["chunk_id"],
                "file": match["source"],
                "page": match["page"],
                "score": round(match["score"], 4),
                "similarity": round(match["similarity"], 4),
                "bm25_score": round(match["bm25_score"], 4),
            }
        )

    # Raw cosine similarity is more honest than a corpus-relative normalized score.
    best_similarity = max((match["similarity"] for match in matches), default=0.0)
    confidence = max(0.0, min(1.0, best_similarity))
    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
        "confidence": round(confidence, 3),
        "search_time": time.perf_counter() - started,
    }


if __name__ == "__main__":
    while question := input(">>> ").strip():
        result = search(question)
        print(result["context"] or "Ничего достаточно релевантного не найдено")
        print(f"confidence={result['confidence']:.3f}")
