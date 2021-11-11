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
database: object = None
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
        res = self.bot.edit_message_media(inline_message_id=self.message_id, **kwargs)
        return self if res else None

    def edit_text(self, **kwargs) -> Optional["InlineMessage"]:
        res = self.bot.edit_message_text(inline_message_id=self.message_id, **kwargs)
        return self if res else None


def set_dispatcher(dp: Dispatcher):
    global dispatcher, database
    dispatcher = dp
    print(dispatcher)
    database = dispatcher.bot_data["conn"]

def get_dispatcher() -> Dispatcher:
    return dispatcher

def get_database() -> object:
    return database


def format_callback_data(
    command: str,
    args: Iterable = [],
    handler_id: str = "core",
    expected_uid: int = None,
) -> str:
    """
Formats callback query data.
Arguments:
    command: `str` - Command to be executed by the handler.
    args: Iterable = [] - An iterable of arguments for the command.
    handler_id: `str` = 'core' - ID of the handler, to which the query will be passed. 
        If set to 'MAIN', the query will be handled in the 'main.py' file.
        If set to 'core', the query will be handled by the core game module.
        Else, 'target_id' is a match ID, to which the query will be passed.
    expected_uid: `int` = None - ID of the user expected to send the query. If None, the query will be accepted from anyone.
Returns: `str` - The resulting callback query data.    
    """
    args = map(lambda x: str(x) if x is not None else "", args)
    return "\n".join(
        [str(expected_uid) if expected_uid is not None else "", handler_id, command, "#".join(args)]
    )


def parse_callback_data(data: str) -> dict[str, Union[str, int, None]]:
    """
Parses data from the callback query, in the app's format.
Arguments:
    data: `str` - The raw callback query data.
Returns: `dict` - Dictoinary with the following fields: 
    'expected_uid': `int` - ID of the user of the user who is expected to send this query. If set to None, the query is accpeted from anyone.
    'handler_id': `str` - ID of the handler, to which the query will be passed. 
        If set to 'MAIN', the query will be handled in the 'main.py` file.
        If set to 'core', the query will be handled by the core game module.
        Else, 'target_id' is a match ID, to which the query will be passed.
    'command': `str` - Command to be executed by the handler.
    'args: `list[int | str]` - Arguments for the command. May be an empty list.
    """
    data = data.split("\n")
    res = {
        "expected_uid": int(data[0]) if data[0] else None,
        "handler_id": data[1],
        "command": data[2],
        "args": [],
    }
    if data[3]:
        for argument in data[3].split("#"):
            res["args"].append(int(argument) if argument.isdigit() else argument)

    return res


def get_tempfile_url(data: bytes, mimetype: str) -> str:
    """
Caches `data` in a file of type specified in `mimetype`, to the images/temp folder and returns a link to the data.
The generated URL is used only once, after it is accessed, the data is deleted from the machine.
Argument:
    data: `bytes` - The data to be returned upon accessing the URL.
    mimetype: `str` - The type of data. Supported types are defined by `mimetypes` built-in module.
Returns: `str` - URL to the data.
    """
    filename = "".join(["temp", uuid.uuid4().hex, mimetypes.guess_extension(mimetype)])
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
Returns: `str` - the resulting ID
"""
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
    callback: TelegramCallback,
    args: tuple = (),
    kwargs: dict = {},
    timeout: int = None,
    is_single: bool = True,
):
    """
Creates the link, which will redirect the user to a chat with the bot,
and automatically send the '/start' command to the bot, the bot will call `f` with the specified `args` and `kwargs`
Arguments:
    dispatcher: `Dispatcher` - dispatcher, associated with the bot.
    callback: `TelegramCallback` - the function to be called. It must accept arguments in the following in the following order: 
        (`Update`, `BoardGameContext`, ...)
    args: `tuple` = () - Additional arguments to be passed to `callback`
    kwargs: `dict` = {} - Keyword arguments to be passed `callback`
    timeout: `int` - time, in seconds, after which the callback will not be called. Default is None, in which case the callback will be cached forever.
    is_single: `bool` = False - if set to True, the callback will be called only once.
Returns: `str` - the resulting link.
"""
    pmsg_id = create_match_id(n=16)

    dispatcher.bot_data["conn"].set(
        f"pm:{pmsg_id}:f", pickle.dumps((callback, args, kwargs)), ex=timeout
    )
    dispatcher.bot_data["conn"].set(
        f"pm:{pmsg_id}:is-single", str(int(is_single)).encode(), ex=timeout
    )

    return helpers.create_deep_linked_url(dispatcher.bot.username, "pmid" + pmsg_id)


def set_result(match_id, results: dict[User, bool]) -> None:
    """
Caches the result of the match and deletes it.
Arguments:
    match_id: `str` - ID of the associated match object. 
    results: `dict[User, bool]` - mapping of `User` object to a boolean, which denotes if the user has won or not.
Return value: None
    """
    for user, is_winner in results.items():
        total_games = int(database.get(f"{user.id}:total") or 0)
        database.set(f"{user.id}:total", str(total_games + 1).encode())
        if is_winner:
            total_wins = int(database.get(f"{user.id}:wins") or 0)
            database.set(f"{user.id}:wins", str(total_wins + 1).encode())

    del dispatcher.bot_data["matches"][match_id]


langtable_cls = lambda obj: DefaultTable(obj, def_f=lambda _, k: k)
langtable = DefaultTable(
    json.load(open("langtable.json"), object_hook=langtable_cls),
    def_f=lambda d, _: d["en"],
)
