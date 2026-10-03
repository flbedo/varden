import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    import pymorphy3
except ImportError:
    pymorphy3 = None


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "summary_model.json"

SHOW_THRESHOLD = 0.35
USE_SEMANTICS = True
SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ME="@flbedo"

MORPH = pymorphy3.MorphAnalyzer() if pymorphy3 else None
XGBOOST_MODEL = None
SEMANTIC_MODEL = None
SEMANTIC_VECTORS = None


KEYWORDS = {
    "event": (
        "мероприятие", "встреча", "конференция", "юбилей",
        "приглашаем", "приходите",
    ),
    "deadline": (
        "дедлайн", "крайний срок", "последний день",
        "срок подачи", "прием заявок", "приём заявок", "срочно",
    ),
    "schedule": (
        "перенос", "перенесли", "перенести", "отмена", "отменили",
        "занятий не будет", "пары не будет", "будет закрыт",
        "будет закрыта", "расписание", "вместо субботы",
    ),
    "education": (
        "пересдача", "комиссия", "лабораторная", "зачет", "зачёт",
        "экзамен", "кафедра", "деканат", "ведомость", "ведомости",
    ),
    "job": (
        "вакансия", "ищем студента", "ищем выпускника",
        "работа", "преподавание", "стажировка", "инвестор", "заявка"
    ),
    "admin": (
        "деканат", "ведомость", "ведомости", "кафедра", "комендатура",
    ),
    "action": (
        "нужно", "необходимо", "принести", "сдать", "забрать",
        "заберите", "написать", "пишите", "зарегистрироваться",
        "дать обратную связь", "дайте обратную связь",
        "отправить", "отправь", "пришли", "помоги", "пожалуйста",
        "ответить", "заполнить", "заполни", "проверить", "проверь",
    ),
    "opportunity": (
        "конкурс", "амбассадор", "амбассадоров", "программа",
        "набор", "заявка", "акселерация", "оффер"
    ),
}


SEMANTIC_PROTOTYPES = {
    "event": "Объявление о мероприятии, встрече или событии.",
    "deadline": "Сообщение о дедлайне или крайнем сроке.",
    "schedule": "Изменение расписания, перенос, отмена или закрытие.",
    "education": "Учебная информация, задание, экзамен или пересдача.",
    "job": "Вакансия, работа, стажировка или преподавание.",
    "admin": "Административная информация от организации.",
    "action": "Сообщение содержит конкретное действие или просьбу.",
    "opportunity": "Конкурс, программа, набор или возможность участия.",
    "noise": "Болтовня, шутка или сообщение без полезных фактов.",
}


DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b")
TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3])[:.\-]\d{2}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USERNAME_RE = re.compile(r"(?<!\w)@[a-zA-Z0-9_]{5,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+7|8)[\s()-]*\d(?:[\s()-]*\d){9}(?!\d)")
WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
LIST_RE = re.compile(r"(?m)^\s*(?:[-–—•▪🔸]|\d+[.)])\s+")


STRUCTURAL_FEATURES = [
    "me_count",
    "word_count",
    "line_count",
    "date_count",
    "time_count",
    "url_count",
    "username_count",
    "phone_count",
    "list_item_count",
    "question_count",
]
KEYWORD_FEATURES = [f"kw_{name}" for name in KEYWORDS]
SEMANTIC_FEATURES = [f"sem_{name}" for name in SEMANTIC_PROTOTYPES]
FEATURE_NAMES = STRUCTURAL_FEATURES + KEYWORD_FEATURES + SEMANTIC_FEATURES


@dataclass
class MessageAnalysis:
    show: bool
    score: float
    tags: list[str]
    source: str
    features: dict[str, float]


@lru_cache(maxsize=20_000)
def lemmatize_word(word: str) -> str:
    if MORPH is None:
        return word
    return MORPH.parse(word)[0].normal_form


def normalize_text(text: str) -> str:
    words = WORD_RE.findall(text.lower().replace("ё", "е"))
    return " ".join(lemmatize_word(word) for word in words)


@lru_cache(maxsize=1)
def normalized_keywords() -> dict[str, tuple[str, ...]]:
    return {
        tag: tuple(normalize_text(phrase) for phrase in phrases)
        for tag, phrases in KEYWORDS.items()
    }


def get_semantic_features(text: str) -> dict[str, float]:
    empty = {name: 0.0 for name in SEMANTIC_FEATURES}

    if not USE_SEMANTICS:
        return empty

    from sentence_transformers import SentenceTransformer


    global SEMANTIC_MODEL, SEMANTIC_VECTORS

    if SEMANTIC_MODEL is None:
        SEMANTIC_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)

    if SEMANTIC_VECTORS is None:
        SEMANTIC_VECTORS = SEMANTIC_MODEL.encode(
            list(SEMANTIC_PROTOTYPES.values()),
            normalize_embeddings=True,
        )

    message_vector = SEMANTIC_MODEL.encode(text, normalize_embeddings=True)

    return {
        f"sem_{name}": float(message_vector @ vector)
        for name, vector in zip(SEMANTIC_PROTOTYPES, SEMANTIC_VECTORS)
    }


def extract_features(text: str) -> dict[str, float]:
    normalized = normalize_text(text)

    features = {
        "me_count": text.count(ME),
        "word_count": len(WORD_RE.findall(text)),
        "line_count": sum(bool(line.strip()) for line in text.splitlines()),
        "date_count": len(DATE_RE.findall(text)),
        "time_count": len(TIME_RE.findall(text)),
        "url_count": len(URL_RE.findall(text)),
        "username_count": len(USERNAME_RE.findall(text)),
        "phone_count": len(PHONE_RE.findall(text)),
        "list_item_count": len(LIST_RE.findall(text)),
        "question_count": text.count("?"),
    }

    for tag, phrases in normalized_keywords().items(): # Счетчик ключевых слов. Например, kw_event = 2, если в тексте встречаются "мероприятие" и "встреча".
        features[f"kw_{tag}"] = sum(
            normalized.count(phrase)
            for phrase in phrases
            if phrase
        )

    features.update(get_semantic_features(text))
    return features


def detect_tags(features: dict[str, float]) -> list[str]:
    tags = []

    for name in KEYWORDS:
        keyword_hit = features[f"kw_{name}"] > 0
        semantic_hit = features[f"sem_{name}"] >= 0.55

        if keyword_hit or semantic_hit:
            tags.append(name.upper())

    return tags


def get_model_probability(features: dict[str, float]) -> float:
    global XGBOOST_MODEL

    import xgboost as xgb

    if XGBOOST_MODEL is None:
        XGBOOST_MODEL = xgb.XGBClassifier()
        XGBOOST_MODEL.load_model(MODEL_PATH)

    row = [[features[name] for name in FEATURE_NAMES]]

    probabilities = XGBOOST_MODEL.predict_proba(row)

    return float(probabilities[0][1])

def reload_model() -> None:
    global XGBOOST_MODEL
    import xgboost as xgb

    XGBOOST_MODEL = xgb.XGBClassifier()
    XGBOOST_MODEL.load_model(MODEL_PATH)


def get_fallback_score(features: dict[str, float]) -> float:
    score = 0.0

    score += min(features["me_count"], 1) * 0.20
    score += min(features["date_count"], 2) * 0.10
    score += min(features["time_count"], 2) * 0.10
    score += min(features["url_count"], 2) * 0.04
    score += min(features["username_count"], 2) * 0.04
    score += min(features["phone_count"], 1) * 0.05
    score += min(features["list_item_count"], 3) * 0.03
    score += min(features["question_count"], 1) * 0.02

    keyword_groups = sum(
        features[f"kw_{name}"] > 0
        for name in KEYWORDS
    )
    keyword_hits = sum(
        features[f"kw_{name}"]
        for name in KEYWORDS
    )

    score += min(keyword_groups, 4) * 0.13
    score += min(keyword_hits, 6) * 0.035

    has_date_or_time = features["date_count"] > 0 or features["time_count"] > 0
    if keyword_groups and has_date_or_time:
        score += 0.15
    if features["kw_deadline"] and features["kw_action"]:
        score += 0.12

    positive_semantics = [
        features[f"sem_{name}"]
        for name in SEMANTIC_PROTOTYPES
        if name != "noise" # "noise" зарезервирован для отрицательной семантики, поэтому его не учитываем в положительных.
    ]

    score += max(positive_semantics, default=0.0) * 0.30
    score -= features["sem_noise"] * 0.12

    return max(0.0, min(score, 1.0))


def analyze_message(text: str) -> MessageAnalysis:
    if not text or not text.strip():
        return MessageAnalysis(
            show=False,
            score=0.0,
            tags=[],
            source="empty",
            features={name: 0.0 for name in FEATURE_NAMES},
        )

    features = extract_features(text)
    tags = detect_tags(features)

    probability = get_model_probability(features)

    if probability is None:
        score = get_fallback_score(features)
        source = "fallback"
    else:
        score = probability
        source = "xgboost"

    

    return MessageAnalysis(
        show=score >= SHOW_THRESHOLD,
        score=round(score, 4),
        tags=tags,
        source=source,
        features=features,
    )


def should_show(text: str) -> bool:
    return analyze_message(text).show


def is_important(text: str) -> bool:
    # Совместимость
    return should_show(text)
