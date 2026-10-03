import os
from pathlib import Path

import easydata as ed

from telethon import TelegramClient, events, functions, types, utils
from telethon.helpers import generate_random_long

from filter import analyze_message, reload_model
from network_config import describe_telethon_proxy, telethon_proxy_from_environment

API_ID = int(os.environ["VCC_TELEGRAM_API_ID"])
API_HASH = os.environ["VCC_TELEGRAM_API_HASH"]

DB_NAME = str(Path(__file__).with_name("varden_data"))
SETTINGS_ID = "settings"
ed.create_database(DB_NAME)


def load_values(name):
    """Загрузить сохранённый в easydata список значений."""
    value = ed.get_item_data(DB_NAME, SETTINGS_ID, name)
    return {item.strip() for item in str(value).split(",") if item.strip()}


def save_values(name, values):
    """Сохранить список значений в easydata."""
    ed.give_item_data(DB_NAME, SETTINGS_ID, name, ",".join(sorted(values)))


# В Telethon "me" означает чат «Избранное». Канал можно указать как @username или числовой Telegram ID.
IMPORTANT_CHAT = ed.get_item_data(DB_NAME, SETTINGS_ID, "important_chat") or "me"
PROCESSED_SOURCES = {
    item.lower().removeprefix("@") for item in load_values("processed_sources")
}
PROCESSED_AUTHORS = load_values("processed_authors")

TELEGRAM_PROXY = telethon_proxy_from_environment()
client = TelegramClient(
    "telegram_selfbot",
    API_ID,
    API_HASH,
    catch_up=True,
    proxy=TELEGRAM_PROXY,
)


def parse_destination(value):
    """Разобрать chat_id[:topic_id] в значения для Telethon."""
    chat_id, separator, topic_id = str(value).partition(":")
    chat_id = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
    return chat_id, int(topic_id) if separator else None


def normalize_source_id(value):
    """Проверить и нормализовать введённый вручную ID источника."""
    source_id, separator, topic_id = str(value).strip().partition(":")
    if not source_id.lstrip("-").isdigit() or int(source_id) == 0:
        raise ValueError("Некорректный ID источника")
    if separator and (not topic_id.isdigit() or int(topic_id) == 0):
        raise ValueError("Некорректный ID топика")

    normalized = str(int(source_id))
    return f"{normalized}:{int(topic_id)}" if separator else normalized


def command(name):
    """Создать декоратор для исходящей self-команды ``.name [текст]``."""
    # pattern — регулярное выражение. Первая группа (.+) содержит аргумент после
    # команды. Например, для «.note купить молоко» это «купить молоко».
    pattern = rf"^:{name}(?:\s+(.+))?$"

    # client.on(...) возвращает декоратор, который регистрирует функцию ниже как
    # обработчик Telethon. outgoing=True пропускает только сообщения, отправленные
    # нашим аккаунтом: чужое входящее сообщение «.note ...» команду не запустит.
    return client.on(events.NewMessage(outgoing=True, pattern=pattern))


def get_topic_id(event):
    """Вернуть ID топика форума или None для обычного чата."""
    if isinstance(getattr(event.message, "action", None), types.MessageActionTopicCreate):
        return event.id

    reply = getattr(event.message, "reply_to", None)
    if not reply:
        return None

    return getattr(reply, "reply_to_top_id", None) or (
        getattr(reply, "reply_to_msg_id", None)
        if getattr(reply, "forum_topic", False)
        else None
    )


async def get_replied_source_id(event):
    """Получить ID чата из сообщения, на которое отправлена команда."""
    message = await event.get_reply_message()
    if message is None:
        return None

    forward = getattr(message, "fwd_from", None)
    if forward:
        # saved_from_peer хранит исходный чат пересланного в «Избранное» сообщения.
        peer = (
            getattr(message, "saved_peer_id", None)
            or getattr(forward, "saved_from_peer", None)
            or getattr(forward, "from_id", None)
        )
        if peer is None:  # Telegram может скрыть источник пересылки настройками приватности.
            return None
    else:
        peer = message.peer_id
    return utils.get_peer_id(peer)


async def get_replied_author_id(event):
    """Получить ID автора сообщения, на которое отправлена команда."""
    message = await event.get_reply_message()
    if message is None:
        return None
    forward = getattr(message, "fwd_from", None)
    if forward:
        peer = getattr(forward, "from_id", None)
        return utils.get_peer_id(peer) if peer else None
    return message.sender_id or (message.chat_id if message.is_private else None)


async def get_source(event):
    """Собрать одинаковое описание источника: личка, группа или канал."""
    chat = await event.get_chat()
    topic_id = get_topic_id(event)
    author_id = event.sender_id or (event.chat_id if event.is_private else None)

    # Чекаем юзернэйм если он есть
    source = {
        "id": event.chat_id,
        "topic_id": topic_id,
        "author_id": author_id,
        "username": getattr(chat, "username", None),
        "title": getattr(chat, "title", None)
        or " ".join(
            filter(
                None,
                [getattr(chat, "first_name", None), getattr(chat, "last_name", None)],
            )
        ),
        "type": "private" if event.is_private else "group" if event.is_group else "channel",
    }

    keys = {str(source["id"]), (source["username"] or "").lower()}
    if topic_id is not None:
        keys.add(f"{source['id']}:{topic_id}")
        if source["username"]:
            keys.add(f"{source['username'].lower()}:{topic_id}")
    # Значение "*" разрешает все источники (условно, если пофиг)
    source["processed"] = "*" in PROCESSED_SOURCES or bool(keys & PROCESSED_SOURCES)
    source["author_processed"] = author_id is not None and (
        "*" in PROCESSED_AUTHORS or str(author_id) in PROCESSED_AUTHORS
    )
    return source


async def send_to_source(source, text):
    """Отправить текст в тот же чат и, если он есть, в тот же топик."""
    return await client.send_message(source["id"], text, reply_to=source["topic_id"])


async def forward_to_destination(chat_id, topic_id, message): # Костыльная функция, потому что Telethon не умеет пересылать в топик форума напрямую.
    """Переслать сообщение, сохранив автора и выбранный forum topic."""
    if topic_id is None:
        return await client.forward_messages(chat_id, message)

    from_peer = await client.get_input_entity(message.peer_id)
    to_peer = await client.get_input_entity(chat_id)
    return await client(
        functions.messages.ForwardMessagesRequest(
            from_peer=from_peer,
            id=[message.id],
            to_peer=to_peer,
            random_id=[generate_random_long()],
            top_msg_id=topic_id,
        )
    )


async def process_message(kind, event, source):
    """Единая точка для своей логики обработки входящих сообщений."""
    if not source["processed"]: return

    analysis = analyze_message(event.raw_text)
    message_text = event.raw_text if len(event.raw_text) < 40 else event.raw_text[:40] + "..."
    print(
        f"[{kind}] {source} message_id={event.id}\n"
        f"show={analysis.show} tags={analysis.tags} source={analysis.source}\n"
        f"score={analysis.score} message={message_text}\n\n"
    )

    ed.give_id_data(
        DB_NAME,
        f"message:{source['id']}:{event.id}",
        {
            "message_id": event.id,
            "source_id": source["id"],
            "topic_id": source["topic_id"],
            "author_id": source["author_id"],
            "text": event.raw_text,
            "reference": -1,
            "tags": ",".join(analysis.tags),
            "score": analysis.score,
            "show": analysis.show,
            "date": getattr(event.message, "date", None),
            "edit_date": getattr(event.message, "edit_date", None),
            "event": kind,
        },
    )

    if not analysis.show: return

    chat_id, topic_id = parse_destination(IMPORTANT_CHAT)
    await forward_to_destination(chat_id, topic_id, event.message)

    # Здесь добавляется ваша логика для разрешённых источников:
    # event.message                 — полный объект сообщения Telethon;
    # event.raw_text                — текст сообщения (пустой у части медиа);
    # await event.get_sender()      — автор сообщения;
    # await send_to_source(source, text) — ответ в этот же чат/топик.


@command("ping")
async def ping(event):
    await event.edit("pong")

@command("test")
async def test(event):
    """Тестовая команда."""
    source_id = await get_replied_source_id(event)
    message = await event.get_reply_message()
    author_id = await get_replied_author_id(event)

    if source_id is None or author_id is None:
        await event.edit("Ответьте командой .test на сообщение")
        return

    analysis = analyze_message(message.raw_text)

    await event.edit(f"Оценка: {analysis.score},\n Видимость: {analysis.show},\n tags: {analysis.tags}\n\n Сообщение: {message.raw_text}")



@command("important")
async def important(event):
    """Переслать сообщение, на которое ответили командой, в IMPORTANT_CHAT."""
    message = await event.get_reply_message() # подсасываем сообщение, на которое ответили командой
    if message is None:
        await event.edit("Ответьте этой командой на нужное сообщение")
        return

    chat_id, topic_id = parse_destination(IMPORTANT_CHAT)
    await forward_to_destination(chat_id, topic_id, message)
    await event.delete()


@command("note")
async def note(event):
    """Пример команды с аргументом: ``.note купить молоко``."""
    # pattern_match — результат сопоставления команды с регулярным выражением из
    # command(). group(1) возвращает первую группу (.+), то есть текст после .note.
    text = event.pattern_match.group(1)
    if not text:
        await event.edit("Использование: .note текст")
        return

    await client.send_message("me", text)
    await event.delete()

@command("relevant")
async def relevant(event):
    """Отметить сообщение, на которое ответили командой, как релевантное (reference=1)."""
    source_id = await get_replied_source_id(event)
    message = await event.get_reply_message()
    author_id = await get_replied_author_id(event)

    if source_id is None or author_id is None:
        await event.edit("Ответьте командой .relevant на сообщение с доступным источником")
        return

    if ed.is_id_exist(DB_NAME, f"message:{author_id}:{event.id}"):
        ed.give_item_data(DB_NAME, f"message:{author_id}:{event.id}", "reference", 1)
    else:
        analysis = analyze_message(event.raw_text)
        ed.give_id_data(
            DB_NAME,
            f"message:{author_id}:{event.id}",
            {
                "message_id": message.id,
                "source_id": message.chat_id,
                "topic_id": getattr(message, "reply_to_msg_id", None),
                "author_id": author_id,
                "text": message.raw_text,
                "reference": 1,
                "tags": ",".join(analysis.tags),
                "score": analysis.score,
                "show": analysis.show,
                "date": getattr(message, "date", None),
                "edit_date": getattr(message, "edit_date", None),
                "event": "new",
            },
        )
    await event.edit(f"Сообщение от автора {author_id}:\n {message.raw_text} \nпомечено как релевантное")

@command("trash")
async def trash(event):
    """Отметить сообщение, на которое ответили командой, как нерелевантное (reference=-1)."""
    source_id = await get_replied_source_id(event)
    message = await event.get_reply_message()
    author_id = await get_replied_author_id(event)

    if source_id is None or author_id is None:
        await event.edit("Ответьте командой .trash на сообщение с доступным источником")
        return

    if ed.is_id_exist(DB_NAME, f"message:{author_id}:{event.id}"):
        ed.give_item_data(DB_NAME, f"message:{author_id}:{event.id}", "reference", 0)
    else:
        analysis = analyze_message(event.raw_text)
        ed.give_id_data(
            DB_NAME,
            f"message:{author_id}:{event.id}",
            {
                "message_id": message.id,
                "source_id": message.chat_id,
                "topic_id": getattr(message, "reply_to_msg_id", None),
                "author_id": author_id,
                "text": message.raw_text,
                "reference": 0,
                "tags": ",".join(analysis.tags),
                "score": analysis.score,
                "show": analysis.show,
                "date": getattr(message, "date", None),
                "edit_date": getattr(message, "edit_date", None),
                "event": "new",
            },
        )
    await event.edit(f"Сообщение от автора {author_id}:\n {message.raw_text} \nпомечено как нерелевантное")

@command("train")
async def train(event):
    """Обучить модель важности на помеченных сообщениях (reference=1 или -1)."""
    from train_model import train_model

    await event.edit("Начинаю обучение модели важности...")
    train_model()
    await event.edit("Обучение модели завершено. Модель сохранена.")
    reload_model()
    await event.edit("Модель важности перезагружена")


@command("source")
async def source_status(event):
    source = await get_source(event)
    await event.edit(
        f"{source['type']} | id={source['id']} | topic={source['topic_id']} | "
        f"processed={source['processed']}"
    )


@command("watch")
async def watch_source(event):
    source_id = await get_replied_source_id(event)
    author_id = await get_replied_author_id(event)
    print(f"watch_source: source_id={source_id} author_id={author_id}")
    topic_id = get_topic_id(event)
    if source_id is None or author_id is None:
        await event.edit("Ответьте командой .watch на сообщение с доступным источником")
        return

    PROCESSED_SOURCES.add(f'{source_id}:{topic_id}' if topic_id else str(source_id))
    PROCESSED_AUTHORS.add(str(author_id))
    save_values("processed_sources", PROCESSED_SOURCES)
    save_values("processed_authors", PROCESSED_AUTHORS)
    await event.edit(f"Добавлены источник {source_id} и автор {author_id}")


@command("addid")
async def add_source_id(event):
    """Добавить ID источника вручную: ``.addid ID[:TOPIC_ID]``."""
    value = event.pattern_match.group(1)
    if not value:
        await event.edit("Использование: .addid ID или .addid ID:TOPIC_ID")
        return

    try:
        source_id = normalize_source_id(value)
    except ValueError:
        await event.edit("Некорректный ID. Использование: .addid ID или .addid ID:TOPIC_ID")
        return

    if source_id in PROCESSED_SOURCES:
        await event.edit(f"Источник {source_id} уже добавлен")
        return

    PROCESSED_SOURCES.add(source_id)
    save_values("processed_sources", PROCESSED_SOURCES)
    await event.edit(f"Добавлен источник {source_id}")


@command("setimportant")
async def set_important_chat(event):
    global IMPORTANT_CHAT
    source_id = await get_replied_source_id(event)
    topic_id = get_topic_id(event)
    if source_id is None:
        await event.edit("Ответьте командой .setimportant на сообщение с доступным источником")
        return

    IMPORTANT_CHAT = f"{source_id}:{topic_id}" if topic_id else str(source_id)
    ed.give_item_data(DB_NAME, SETTINGS_ID, "important_chat", IMPORTANT_CHAT)
    await event.edit(f"Важным назначен источник {IMPORTANT_CHAT}")



@client.on(events.NewMessage(incoming=True))
async def incoming_message(event):
    await process_message("new", event, await get_source(event))


@client.on(events.MessageEdited(incoming=True))
async def edited_message(event):
    await process_message("edited", event, await get_source(event))


# @client.on(events.MessageDeleted())
# async def deleted_message(event):
#     # TODO: потом сделать буфер удалённых сообщений, чтобы можно было их восстановить командой
#     source_id = event.chat_id
#     processed = "*" in PROCESSED_SOURCES or str(source_id) in PROCESSED_SOURCES
#     print(f"[deleted] source_id={source_id} processed={processed} message_ids={event.deleted_ids}")


if __name__ == "__main__":
    print("Selfbot запущен. Команды: .ping, .important, .note, .source, .watch, .addid, .setimportant, .trash, .train")
    print(f"Подключение к Telegram: {describe_telethon_proxy(TELEGRAM_PROXY)}")
    client.start()

    client.run_until_disconnected()
