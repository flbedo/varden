"""Секретарь, который разбирает сообщения через локальную Ollama."""

import json
from datetime import datetime, timezone

import httpx

import easydata as ed


OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODELS = {
    "fast": "qwen3:0.6b",
    "middle": "qwen2.5-coder:3b",
    "quality": "llama3.1:8b",
}
DEFAULT_OLLAMA_MODE = "fast"
CONTEXT_MESSAGES = 2
MIN_CONTEXT_SCORE = 0.35
CHUNK_SIZE = 30
MAX_CHUNK_CHARS = 12_000


SUMMARY_PROMPT = """Ты — внимательный секретарь. Ты получаешь сообщения одного автора
и раскладываешь полезную информацию по трём разделам:
- tasks — конкретные задачи, просьбы и действия, которые нужно выполнить;
- events — встречи, занятия, звонки и другие мероприятия с датой или временем;
- messages — важные факты, решения, изменения и объявления, которые нужно знать.

Поля show и score выставлены другой моделью и могут быть ошибочными.
show=true не гарантирует полезность сообщения, а show=false не означает,
что сообщение нужно игнорировать. Убери реакции, смех, бессодержательные
ответы и мусор. Учитывай исправления и уточнения внутри последовательности.
Не выдумывай информацию. Не добавляй приветствия,
обращения, комментарии от себя и вводные фразы, которых нет в сообщениях.
Сохраняй даты, время, сроки, места, ссылки и другие важные детали. Учитывай дату
самого сообщения, когда в тексте встречаются слова вроде «сегодня» и «завтра».
Один факт помещай только в самый подходящий раздел, без повторов.

Ответь JSON-объектом вида:
{
  "tasks": [{"text": "Сдать отчёт до пятницы", "message_ids": [1]}],
  "events": [{"text": "Встреча в пятницу в 15:00", "message_ids": [2]}],
  "messages": [{"text": "Дедлайн перенесли", "message_ids": [3, 4]}]
}
message_ids должны содержать только ID сообщений, на которых основан пункт.
Если в разделе ничего нет, верни для него пустой список.

Сообщения:
"""

CATEGORIES = ("tasks", "events", "messages")


def normalize_datetime(value: datetime | str) -> datetime:
    """Привести даты из БД и границы периода к UTC."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def load_messages(db_name: str, start: datetime, end: datetime) -> list[dict]:
    """Загрузить сохранённые сообщения за включительный промежуток времени."""
    messages = []
    start = normalize_datetime(start)
    end = normalize_datetime(end)

    for message in ed.get_ids_data(db_name, "message:").values():
        date = message.get("date") if message else None
        if date is None:
            continue
        try:
            date = normalize_datetime(date)
        except (TypeError, ValueError):
            continue
        if start <= date <= end:
            messages.append({**message, "date": date})

    messages.sort(key=lambda message: message["date"])
    return messages


def select_messages(messages: list[dict]) -> list[dict]:
    """Оставить важные сообщения и ближайший контекст того же чата и темы."""
    groups = {}
    selected_indexes = set()

    for index, message in enumerate(messages):
        key = (message.get("source_id"), message.get("topic_id"))
        groups.setdefault(key, []).append(index)

    for indexes in groups.values():
        for position, index in enumerate(indexes):
            message = messages[index]
            score = float(message.get("score") or 0)
            if not message.get("show") and score < MIN_CONTEXT_SCORE:
                continue

            first = max(0, position - CONTEXT_MESSAGES)
            last = min(len(indexes), position + CONTEXT_MESSAGES + 1)
            selected_indexes.update(indexes[first:last])

    return [message for index, message in enumerate(messages) if index in selected_indexes]


def format_messages(messages: list[dict]) -> str:
    """Подготовить сообщения в простом читаемом формате для модели."""
    parts = []

    for message in messages:
        show = str(bool(message.get("show"))).lower()
        score = float(message.get("score") or 0)
        date = message.get("date")
        if isinstance(date, datetime):
            date = date.isoformat()
        date_part = f" date={date}" if date else ""
        parts.append(
            f'[id={message["message_id"]}{date_part} show={show} score={score:.2f}]\n'
            f'{message.get("text") or ""}'
        )

    return "\n\n".join(parts)


def message_link(message: dict) -> str | None:
    """Построить Telegram-ссылку на сообщение группы или канала."""
    username = str(message.get("source_username") or "").removeprefix("@")
    source_id = str(message.get("source_id") or "")

    if username:
        chat = username
    elif source_id.startswith("-100"):
        chat = f"c/{source_id[4:]}"
    else:
        return None

    topic_id = message.get("topic_id")
    message_id = message["message_id"]
    if topic_id:
        return f"https://t.me/{chat}/{topic_id}/{message_id}"
    return f"https://t.me/{chat}/{message_id}"


def chunk_messages(messages: list[dict]) -> list[list[dict]]:
    """Разбить вход по числу сообщений и размеру промпта."""
    chunks = []
    chunk = []
    chunk_chars = 0

    for message in messages:
        formatted_size = len(format_messages([message]))
        separator_size = 2 if chunk else 0
        if chunk and (
            len(chunk) >= CHUNK_SIZE
            or chunk_chars + separator_size + formatted_size > MAX_CHUNK_CHARS
        ):
            chunks.append(chunk)
            chunk = []
            chunk_chars = 0

        chunk.append(message)
        chunk_chars += (2 if len(chunk) > 1 else 0) + formatted_size

    if chunk:
        chunks.append(chunk)
    return chunks


def ask_ollama(prompt: str, model: str | None = None) -> dict:
    """Отправить один JSON-запрос локальной Ollama."""
    model = model or OLLAMA_MODELS[DEFAULT_OLLAMA_MODE]
    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()
    return json.loads(response.json()["response"])


def clean_result(result: dict, allowed_ids: set) -> dict:
    """Очистить пункты и убрать ID, которых не было во входных сообщениях."""
    cleaned = {category: [] for category in CATEGORIES}

    for category in CATEGORIES:
        items = result.get(category, [])
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            text = str(item.get("text") or "").strip()
            message_ids = []
            for raw_message_id in item.get("message_ids", []):
                try:
                    message_id = int(raw_message_id)
                except (TypeError, ValueError):
                    continue
                if message_id in allowed_ids and message_id not in message_ids:
                    message_ids.append(message_id)

            if text and message_ids:
                cleaned[category].append({"text": text, "message_ids": message_ids})

    return cleaned


def summarize_author(messages: list[dict], model: str | None = None) -> dict:
    """Разобрать сообщения одного автора как секретарь."""
    if not messages:
        return {category: [] for category in CATEGORIES}

    summaries = []
    allowed_ids = {message["message_id"] for message in messages}

    for chunk in chunk_messages(messages):
        result = ask_ollama(SUMMARY_PROMPT + format_messages(chunk), model)
        chunk_ids = {message["message_id"] for message in chunk}
        summaries.append(clean_result(result, chunk_ids))

    if len(summaries) == 1:
        return summaries[0]

    prompt = """Ты — секретарь. Объедини частичные результаты по сообщениям
одного автора. Не добавляй новых фактов, удали повторы и учти поздние
исправления. Верни JSON с тремя списками tasks, events и messages. Каждый пункт
должен иметь вид {"text": "полезный факт", "message_ids": [1, 2]}.
Сохрани только ID, на которых основан итоговый пункт.

Частичные результаты:
""" + json.dumps(summaries, ensure_ascii=False)
    return clean_result(ask_ollama(prompt, model), allowed_ids)


def summarize_messages(messages: list[dict], model: str | None = None) -> dict[str, list]:
    """Вернуть общий список задач, мероприятий и важных сообщений."""
    by_author = {}

    for message in messages:
        author_id = message.get("author_id")
        if author_id is not None:
            by_author.setdefault(author_id, []).append(message)

    result = {category: [] for category in CATEGORIES}
    for author_id, author_messages in by_author.items():
        selected = select_messages(author_messages)
        author_result = summarize_author(selected, model)
        messages_by_id = {}
        for message in author_messages:
            messages_by_id.setdefault(message["message_id"], message)

        for category in CATEGORIES:
            for item in author_result[category]:
                message_links = []
                for message_id in item["message_ids"]:
                    source_message = messages_by_id.get(message_id)
                    link = message_link(source_message) if source_message else None
                    if link:
                        message_links.append({"message_id": message_id, "url": link})

                result_item = {**item, "author_id": author_id}
                if message_links:
                    result_item["message_links"] = message_links
                result[category].append(result_item)

    return result
