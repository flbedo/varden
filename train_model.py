"""Обучение модели важности на примерах и помеченных сообщениях."""

from pathlib import Path

import easydata as ed
import xgboost as xgb

from filter import FEATURE_NAMES, MODEL_PATH, extract_features


DB_NAME = str(Path(__file__).with_name("varden_data"))

SAMPLES = [
    ("Срочно отправь отчет сегодня", 1),
    ("Завтра встреча команды в 14:30", 1),
    ("Обязательно заполни форму до завтра", 1),
    ("Дедлайн сегодня, нужно закончить задачу", 1),
    ("Важный созвон в 10:00", 1),
    ("Пожалуйста, пришли документы сегодня", 1),
    ("Экзамен 12.10 в 09:00", 1),
    ("Немедленно проверь сервер", 1),
    ("Собрание отдела завтра", 1),
    ("Нужно ответить клиенту до вечера", 1),
    ("Привет, как дела?", 0),
    ("Спасибо", 0),
    ("Хорошая погода", 0),
    ("Посмотри смешное видео", 0),
    ("Я уже пообедал", 0),
    ("Доброе утро", 0),
    ("Может быть", 0),
    ("Интересная новость", 0),
    ("До встречи", 0),
    ("Отлично", 0),
    ("хахахахахах", 0),
]


def load_training_samples() -> list[tuple[str, int]]:
    """Дополнить базовые примеры сообщениями с ручной оценкой из БД."""
    samples = list(SAMPLES)
    ed.create_database(DB_NAME)

    for message_id in ed.ids(DB_NAME):
        if not message_id.startswith("message:"):
            continue

        message = ed.get_id_data(DB_NAME, message_id) or {}
        label = message.get("reference")    
        text = message.get("text")
        if label in (0, 1) and isinstance(text, str) and text.strip():
            samples.append((text, label))

    return samples


def train_model() -> None:
    samples = load_training_samples()
    rows = []
    for text, _ in samples:
        extracted = extract_features(text)
        rows.append([extracted[name] for name in FEATURE_NAMES])

    labels = [label for _, label in samples]
    data = xgb.DMatrix(rows, label=labels, feature_names=FEATURE_NAMES)

    model = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 3,
            "eta": 0.2,
        },
        data,
        num_boost_round=30,
    )
    model.save_model(MODEL_PATH)
    print(f"Модель обучена на {len(samples)} примерах и сохранена: {MODEL_PATH}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
