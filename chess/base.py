import os
import json
from typing import Any, Callable, Iterable, Optional, Union
import uuid
import mimetypes
import pickle
import random
from telegram.ext import Dispatcher, CallbackContext
from telegram.utils import helpers
from telegram import Bot, Update, User

TelegramCallback = Callable[[Update, CallbackContext], None]
database = None
dispatcher: Dispatcher = None

class DefaultTable(dict):
    def __init__(self, *args, def_f: Callable[["DefaultTable", Any], Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.def_f = def_f

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as exc:
            if self.def_f:
                return self.def_f(self, key)
            else:
                raise exc


class InlineMessage:
    def __init__(self, message_id: str, bot: Bot):
        self.message_id = message_id
        self.bot = bot

    def edit_caption(self, **kwargs) -> Optional["InlineMessage"]:
        res = self.bot.edit_message_caption(inline_message_id=self.message_id, **kwargs)
        return self if res else None

    def edit_media(self, **kwargs) -> Optional["InlineMessage"]:
        res = self.bot.edit_message_media(inline_message_id=self.message_id, **kwargs)
        return self if res else None

    def edit_text(self, **kwargs) -> Optional["InlineMessage"]:
        res = self.bot.edit_message_media(inline_message_id=self.message_id, **kwargs)
        return self if res else None

    def edit_text(self, **kwargs) -> Optional["InlineMessage"]:
        res = self.bot.edit_message_text(inline_message_id=self.message_id, **kwargs)
        return self if res else None


def set_dispatcher(dp: Dispatcher):
    global dispatcher, database
    dispatcher = dp
    database = dispatcher.bot_data["conn"]


def format_callback_data(
    command: str,
    args: Iterable = [],
    handler_id: str = "core",
    expected_uid: int = None,
) -> str:
    args = map(lambda x: str(x) if x is not None else "", args)
    return "\n".join(
        [str(expected_uid) if expected_uid else "", handler_id, command, "#".join(args)]
    )


def parse_callback_data(data: str) -> dict[str, Union[str, int, None]]:
    data = data.split("\n")
    res = {
        "expected_uid": int(data[0]) if data[0] else None,
        "target_id": data[1],
        "command": data[2],
        "args": [],
    }
    if data[3]:
        for argument in data[3].split("#"):
            res["args"].append(int(argument) if argument.isdigit() else argument)

    return res


def get_tempfile_url(data: bytes, mimetype: str) -> str:
    filename = "".join(["temp", uuid.uuid4().hex, mimetypes.guess_extension(mimetype)])
    open(os.path.join("images", "temp", filename), "wb").write(data)
    return "/".join(
        [os.environ["HOST_URL"], os.environ["BOT_TOKEN"], "dynamic", filename]
    )


def get_file_url(filename: str) -> str:
    return "/".join(
        [os.environ["HOST_URL"], os.environ["BOT_TOKEN"], "static", filename]
    )


def create_match_id(n=8) -> str:
    return "".join(
        [
            random.choice(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            )
            for _ in range(n)
        ]
    )


def set_pending_message(
    dispatcher: Dispatcher,
    f: TelegramCallback,
    args: tuple = (),
    timeout: int = None,
    is_single: bool = True,
):
    pmsg_id = create_match_id(n=16)

    dispatcher.bot_data["conn"].set(
        f"pm:{pmsg_id}:f", pickle.dumps((f, args)), ex=timeout
    )
    dispatcher.bot_data["conn"].set(
        f"pm:{pmsg_id}:is-single", str(int(is_single)).encode(), ex=timeout
    )

    return helpers.create_deep_linked_url(dispatcher.bot.username, "pmid" + pmsg_id)


def set_result(match, user_results: dict[User, bool]):
    for user, is_winner in user_results.items():
        total_games = int(database.get(f"{user.id}:total") or 0)
        database.set(f"{user.id}:total", str(total_games + 1).encode())
        if is_winner:
            total_wins = int(database.get(f"{user.id}:wins") or 0)
            database.set(f"{user.id}:wins", str(total_wins + 1).encode())

    del dispatcher.bot_data["matches"][match.id]


langtable_cls = lambda obj: DefaultTable(obj, def_f=lambda _, k: k)
langtable = DefaultTable(
    json.load(open("langtable.json"), object_hook=langtable_cls),
    def_f=lambda d, _: d["en"],
)
