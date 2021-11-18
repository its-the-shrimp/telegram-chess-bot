import os
import json
from typing import Any, Callable, Iterator, Optional, TypedDict, cast, Generator
import uuid
import mimetypes
import pickle
import random
from telegram.ext import Dispatcher, CallbackContext
from telegram.utils import helpers
from telegram import Bot, Update, User, InlineKeyboardMarkup, InlineKeyboardButton
import redis

TextCommand = Callable[[Update, "BoardGameContext"], None]
KeyboardCommand = Callable[[Update, "BoardGameContext", list[str]], None]
_UserData = TypedDict("_UserData", {"is_anon": bool, "lang_code": str, "total": int, "wins": int})
database: "RedisInterface"
dispatcher: Dispatcher

class RedisInterface(redis.Redis):
    bot: Bot

    def _fetch_matches(
        self, decoder: Callable[[bytes, Dispatcher, str], Any], dispatcher: Dispatcher
    ) -> Generator[tuple[str, Any], None, None]:
        for key in self.scan_iter(match="match:*"):
            mid: str = key.split(b":")[1].decode()
            yield mid, decoder(cast(bytes, self.get(key)), dispatcher, mid)

    def _flush_matches(self, matches: dict[str, object]) -> None:
        for id, matchobj in matches.items():
            self.set(f"match:{id}", bytes(matchobj))    # type: ignore

    def get_pending_message(
        self, pmid: str
    ) -> Optional[tuple[Callable[[Update, "BoardGameContext", tuple], None], tuple]]:
        raw_f = self.get(f"pm:{pmid}:f")
        if int(cast(bytes, self.get(f"pm:{pmid}:is-single")).decode()) and raw_f:
            self.delete(f"pm:{pmid}:f", f"pm:{pmid}:is-single")
            return pickle.loads(raw_f)
        return None

    def set_pending_message(
        self,
        callback: Callable[[Update, "BoardGameContext", tuple], None],
        args: tuple = (),
        timeout: int = None,
        is_single: bool = True,
    ):
        """
        Creates the link, which will redirect the user to a chat with the bot,
        and automatically send the '/start' command to the bot, the bot will call `f` with the specified `args` and `kwargs`
        Arguments:
            dispatcher: `Dispatcher` - dispatcher, associated with the bot.
            callback: `TextCommand` - the function to be called. It must accept arguments in the following in the following order:
                (`Update`, `BoardGameContext`, ...)
            args: `tuple` = () - Additional arguments to be passed to `callback`
            kwargs: `dict` = {} - Keyword arguments to be passed `callback`
            timeout: `int` - time, in seconds, after which the callback will not be called. Default is None, in which case the callback will be cached forever.
            is_single: `bool` = False - if set to True, the callback will be called only once.
        Returns: `str` - the resulting link."""
        pmsg_id = create_match_id(n=16)

        self.set(
            f"pm:{pmsg_id}:f", pickle.dumps((callback, args)), ex=timeout
        )
        self.set(
            f"pm:{pmsg_id}:is-single", str(int(is_single)).encode(), ex=timeout
        )

        return helpers.create_deep_linked_url(self.bot.username, "pmid" + pmsg_id)

    def get_name(self, user: User) -> str:
        """
        If user has enabled anonymous mode, returns "player(ID of the user)",
        otherwise returns his username, or full name, if no username is available.
        Arguments:
            user: `User` - User whose name needs to bee queried.
        Returns: `str` - Name of the user.
        """
        if self.exists(f"{user.id}:isanon"):
            return f"player{user.id}"
        else:
            return user.name

    def create_invite(self, invite_id: str, user: User, options: dict) -> None:
        self.set(
            f"invite:{invite_id}",
            json.dumps({"from_user": user.to_dict(), "options": options}).encode(),
            ex=1800,
        )
        return None

    def get_invite(self, invite_id: str) -> Optional[dict]:
        raw = self.get(f"invite:{invite_id}")
        if raw:
            self.delete(f"invite:{invite_id}")
            res = json.loads(raw.decode())
            res["from_user"] = User.de_json(res["from_user"], self.bot)
            return res
        return None

    def set_anon_mode(self, user: User, value: bool) -> None:
        if value:
            self.set(f"{user.id}:isanon", b"1")
        else:
            self.delete(f"{user.id}:isanon")

    def get_user_ids(self) -> Generator[int, None, None]:
        for key in self.scan_iter(match="*:lang"):
            key = key.decode()
            yield int(key[: key.find(":")])

    def get_langcodes_stats(self) -> dict[str, int]:
        counter: dict[str, int] = {}
        for key in self.keys(pattern="*:lang"):
            key = cast(bytes, self[key]).decode()
            if key in counter:
                counter[key] += 1
            else:
                counter[key] = 1

        return counter

    def get_user_data(self, user_id: int) -> _UserData:
        return _UserData(
            lang_code=(self.get(f"{user_id}:lang") or b"en").decode(),
            is_anon=bool(self.exists(f"{user_id}:isanon")),
            total=int(self.get(f"{user_id}:total") or 0),
            wins=int(self.get(f"{user_id}:wins") or 0),
        )

    def del_user_data(self, user_id: int) -> bool:
        return bool(
            self.delete(
                f"{user_id}:lang",
                f"{user_id}:isanon",
                f"{user_id}:total",
                f"{user_id}:wins",
            )
        )


class OptionValue:
    def __init__(
        self,
        value: str,
        condition: Callable[[dict[str, Optional[str]]], bool] = None,
        var_name: str = None,
    ):
        self.value = value
        self.is_available = condition or (lambda _: True)
        self.var_name = var_name

    def __repr__(self):
        return f"<OptionValue({self.value}{ (', ' + self.var_name) if self.var_name is not None else '' })>"

    def __eq__(self, other):
        return isinstance(self, other.__class__) and self.value == other.value


class MenuOption:
    @classmethod
    def from_dict(cls, name: str, obj: dict) -> "MenuOption":
        return cls(
            name,
            [
                OptionValue(k, condition=v, var_name=obj.get("vars", {}).get(k))
                for k, v in obj["values"].items()
            ],
            condition=obj.get("condition"),
        )

    def __init__(
        self,
        name: str,
        values: list[OptionValue],
        condition: Callable[[dict[str, Optional[str]]], bool] = None,
    ):
        self.name = name
        self.values = values
        self.is_available = condition or (lambda _: True)

    def __repr__(self):
        return f"<MenuOption({self.name}); {len(self.values)} values>"

    def __iter__(self) -> Iterator[OptionValue]:
        return self.values.__iter__()

    def available_values(self, options: dict[str, Optional[str]]) -> list[str]:
        if not self.is_available(options):
            return []
        return [i.value for i in self.values if i.is_available(options)]


class MenuFormatter:
    @classmethod
    def from_dict(cls, obj: dict) -> "MenuFormatter":
        return cls([MenuOption.from_dict(*args) for args in obj.items()])

    def __init__(self, options: list[MenuOption]):
        self.options: list[MenuOption] = options
        self.defaults: dict[str, Optional[str]] = {}

        for option in self:
            if not self.defaults:
                self.defaults[option.name] = option.values[0].value
            else:
                self.defaults[option.name] = self.get_default_value(
                    option.name, self.defaults
                )

    def __iter__(self) -> Iterator[MenuOption]:
        return self.options.__iter__()

    def __getitem__(self, key: str) -> MenuOption:
        for option in self:
            if option.name == key:
                return option

        raise KeyError(key)

    @staticmethod
    def is_valid(keyboard: Optional[InlineKeyboardMarkup]) -> bool:
        return (
            keyboard is not None
            and len(keyboard.inline_keyboard[-1]) == 2
            and all([len(i) == 3 for i in keyboard.inline_keyboard[:-1]])
        )

    def get_value(self, key: str, index: int) -> str:
        lookup_list = [value.value for value in self[key]]
        return lookup_list[index]

    def get_index(self, key: str, value: str) -> int:
        lookup_list = [value.value for value in self[key]]
        return lookup_list.index(value)

    def format_notes(self, context: "BoardGameContext", options: dict[str, Optional[str]] = None):
        options = options or self.defaults
        notes = []
        if options["mode"] == "online":
            notes.append(
                context.langtable["main:queue-len"].format(
                    n=len(context.bot_data["queue"])
                )
            )
        for value in options.values():
            notes += (
                [context.langtable[f"{value}:note"]]
                if f"{value}:note" in context.langtable
                else []
            )
        notes.extend(("", context.langtable["main:game-setup"]))

        return "\n".join(notes)

    def get_default_value(self, name: str, options: dict[str, Optional[str]]) -> Optional[str]:
        try:
            return self[name].available_values(options)[0]
        except LookupError:
            return None

    def get_variables(self, options: dict[str, str]) -> list[str]:
        res = []
        for k, value_name in options.items():
            res += [
                v.var_name
                for v in self[k]
                if v.var_name is not None
                and v.value == value_name
                and v.var_name not in options
            ]

        return res

    def decode(self, keyboard: InlineKeyboardMarkup) -> dict[str, str]:
        res = {}
        for column in keyboard.inline_keyboard[:-1]:
            column_data = CallbackData.decode(cast(str, column[1].callback_data)).args
            res[column_data[0]] = self.get_value(column_data[0], int(column_data[1]))

        return res

    def encode(
        self, user: User, indexes: dict[str, Optional[str]] = None
    ) -> InlineKeyboardMarkup:
        res = []
        indexes = indexes or self.defaults

        for option in self:
            if option.is_available(indexes):
                if option.name in indexes:
                    chosen = indexes[option.name]
                else:
                    chosen = cast(str, self.get_default_value(option.name, indexes))
                    indexes[option.name] = chosen
                res.append(
                    [
                        InlineKeyboardButton(
                            text="◀️",
                            callback_data=str(CallbackData(
                                "PREV",
                                args=[option.name],
                                expected_uid=user.id,
                                handler_id="MAIN",
                            )),
                        ),
                        InlineKeyboardButton(
                            text=langtable[user.language_code][chosen],
                            callback_data=str(CallbackData(
                                "DESC",
                                args=[option.name, str(self.get_index(option.name, cast(str, chosen)))],
                                handler_id="MAIN",
                                expected_uid=user.id,
                            )),
                        ),
                        InlineKeyboardButton(
                            text="▶️",
                            callback_data=str(CallbackData(
                                "NEXT",
                                args=[option.name],
                                handler_id="MAIN",
                                expected_uid=user.id,
                            )),
                        ),
                    ]
                )
            else:
                if option.name in indexes:
                    del indexes[option.name]

        inline_query = self.tg_encode(indexes) if indexes["mode"] == "invite" else None

        res.append(
            [
                InlineKeyboardButton(
                    text=langtable[user.language_code]["main:cancel-button"],
                    callback_data=str(CallbackData(
                        "REMOVE_MENU", handler_id="MAIN", expected_uid=user.id
                    )),
                ),
                InlineKeyboardButton(
                    text=langtable[user.language_code]["main:play-button"],
                    switch_inline_query=inline_query,
                    callback_data=None
                    if inline_query is not None
                    else str(CallbackData(
                        "PLAY", handler_id="MAIN", expected_uid=user.id
                    )),
                ),
            ]
        )

        return InlineKeyboardMarkup(res)

    def prettify(self, options: dict[str, Optional[str]], lang_code: str) -> str:
        locale = langtable[lang_code]

        return "\n".join(
            [
                locale["main:options-title"],
                ", ".join([locale[v] for k, v in options.items() if k != "mode" and v is not None]),
            ]
        )

    def tg_encode(self, indexes: dict[str, Optional[str]]) -> str:
        return "&".join(["=".join((k, v)) for k, v in indexes.items() if k != "mode" and v is not None])

    def tg_decode(self, src: str) -> dict[str, Optional[str]]:
        indexes: dict[str, Optional[str]] = dict(cast(tuple[str, str], token.split("=")) for token in src.split("&") if "=" in token)
        for option in self:
            is_available = option.is_available(indexes)
            if option.name not in indexes and is_available:
                indexes[option.name] = cast(str, self.get_default_value(option.name, indexes))
            elif option.name in indexes and not is_available:
                del indexes[option.name]

        return indexes


class BoardGameContext(CallbackContext):
    menu: MenuFormatter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.langtable: DefaultTable = None
        RedisInterface.bot = self.dispatcher.bot

    @classmethod
    def from_update(
        cls, update: object, dispatcher: Dispatcher
    ) -> "BoardGameContext":
        self = super().from_update(update, dispatcher)
        if isinstance(update, Update) and isinstance(update.effective_user, User):
            self.langtable = langtable[update.effective_user.language_code]
            self.db.set(
                f"{update.effective_user.id}:lang",
                cast(str, update.effective_user.language_code).encode(),
            )

        return self

    @property
    def db(self) -> RedisInterface:
        return self.dispatcher.bot_data["conn"]


class CallbackData:
    expected_uid: Optional[int]
    handler_id: str
    command: str
    args: list[str]

    def __init__(self, command: str, expected_uid: Optional[int] = None, handler_id: str = "MAIN", args: list[str] = []):
        self.command = command
        self.expected_uid = expected_uid
        self.handler_id = handler_id
        self.args = args

    def __str__(self):
        return "\n".join([
            str(self.expected_uid) if self.expected_uid is not None else "",
            self.handler_id,
            self.command,
            "#".join(self.args)
        ])

    @classmethod
    def decode(cls, src: str) -> "CallbackData":
        data = src.split("\n")
        res = cls(
            expected_uid=int(data[0]) if data[0] else None,
            handler_id=data[1],
            command=data[2],
            args=data[3].split("#") if data[3] else []
        )

        return res


class DefaultTable(dict):
    def __init__(
        self, *args, def_f: Callable[["DefaultTable", Any], Any] = None, **kwargs
    ):
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
    """An adapter class for inline messages, i.e. messages sent via an inline query."""

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
        res = self.bot.edit_message_text(inline_message_id=self.message_id, **kwargs)
        return self if res else None


def set_dispatcher(dp: Dispatcher):
    global dispatcher, database
    dispatcher = dp
    database = dispatcher.bot_data["conn"]


def get_dispatcher() -> Dispatcher:
    return dispatcher


def get_database() -> RedisInterface:
    return database


def get_tempfile_url(data: bytes, mimetype: str) -> str:
    """
    Caches `data` in a file of type specified in `mimetype`, to the images/temp folder and returns a link to the data.
    The generated URL is used only once, after it is accessed, the data is deleted from the machine.
    Argument:
        data: `bytes` - The data to be returned upon accessing the URL.
        mimetype: `str` - The type of data. Supported types are defined by `mimetypes` built-in module.
    Returns: `str` - URL to the data.
    """
    extension = mimetypes.guess_extension(mimetype)
    assert extension is not None, "Unknown file format"
    filename = "".join(["temp", uuid.uuid4().hex, extension])
    open(os.path.join("images", "temp", filename), "wb").write(data)
    return "/".join(
        [os.environ["HOST_URL"], os.environ["BOT_TOKEN"], "dynamic", filename]
    )


def get_file_url(filename: str) -> str:
    """
    Creates a link to the file located in the server's images/static folder.
    Argument:
        filename: `str` - Name of the file.
    Returns: `str` - A link to the file.
    """
    return "/".join(
        [os.environ["HOST_URL"], os.environ["BOT_TOKEN"], "static", filename]
    )


def create_match_id(n=8) -> str:
    """
    Creates a unique ID from URL safe Base64 characters.
    Arguments:
        n: `int` = 8 - Amount of characters in the ID.
    Returns: `str` - the resulting ID"""
    return "".join(
        [
            random.choice(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            )
            for _ in range(n)
        ]
    )


def set_result(match_id, results: dict[User, bool]) -> None:
    """
    Caches the result of the match and deletes it.
    Arguments:
        match_id: `str` - ID of the associated match object.
        results: `dict[User, bool]` - mapping of `User` object to a boolean, which denotes if the user has won or not.
    Return value: None
    """
    for user, is_winner in results.items():
        total_games = int(database.get(f"{user.id}:total") or 0)    # type: ignore
        database.set(f"{user.id}:total", str(total_games + 1).encode())    # type: ignore
        if is_winner:
            total_wins = int(database.get(f"{user.id}:wins") or 0)   # type: ignore
            database.set(f"{user.id}:wins", str(total_wins + 1).encode())    # type: ignore

    del dispatcher.bot_data["matches"][match_id]
    database.delete(f"match:{match_id}")    # type: ignore


langtable_cls = lambda obj: DefaultTable(obj, def_f=lambda _, k: k)
langtable = DefaultTable(
    json.load(open("langtable.json"), object_hook=langtable_cls),
    def_f=lambda d, _: d["en"],
)
