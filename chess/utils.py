import collections
import os
from typing import Optional
import json
from typing import Callable, List, Union, Optional
from telegram import Bot
import uuid
import mimetypes
import random

FENSYMBOLS = {
    "k": "King",
    "q": "Queen",
    "r": "Rook",
    "b": "Bishop",
    "n": "Knight",
    "p": "Pawn",
}
ENGINE_FILENAME = "./stockfish_14_x64"
STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BoardPoint = collections.namedtuple("BoardPoint", ("column", "row"), module="chess")


class DefaultTable(dict):
    def __init__(self, *args, def_f: Callable = None, **kwargs):
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


class InlineMessageAdapter:
    def __init__(self, message_id: str, bot: Bot):
        self.message_id = message_id
        self.bot = bot

    def edit_caption(self, caption: str, **kwargs) -> "InlineMessageAdapter":
        self.bot.edit_message_caption(
            caption=caption, inline_message_id=self.message_id, **kwargs
        )
        return self

    def edit_media(self, **kwargs) -> "InlineMessageAdapter":
        self.bot.edit_message_media(inline_message_id=self.message_id, **kwargs)
        return self


def format_callback_data(
    args: List[Union[str, int]],
    handler_id: str = "module",
    expected_user_id: int = None,
):
    return f"{expected_user_id or ''}\n{handler_id}\n{ '#'.join(args) }"


def get_tempfile_url(data: bytes, mimetype: str) -> str:
    filename = "".join([
        "temp",
        uuid.uuid4().hex,
        mimetypes.guess_extension(mimetype)
    ])
    open(os.path.join("images", "temp", filename), "wb").write(data)
    return "/".join([os.environ["HOST_URL"], os.environ["BOT_TOKEN"], "dynamic", filename])


def get_file_url(filename: str) -> str:
    return "/".join([os.environ["HOST_URL"], os.environ["BOT_TOKEN"], "static", filename])


def create_match_id(n=8) -> str:
    return "".join(
        [
            random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-+")
            for _ in range(n)
        ]
    )


langtable_cls = lambda obj: DefaultTable(obj, def_f=lambda _, k: k)
langtable = DefaultTable(
    json.load(open("langtable.json"), object_hook=langtable_cls),
    def_f=lambda d, _: d["en"],
)


def decode_pos(pos: str) -> Optional[BoardPoint]:
    return BoardPoint(ord(pos[0]) - 97, int(pos[1]) - 1) if pos != "-" else None


def encode_pos(pos: BoardPoint) -> str:
    return chr(pos.column + 97) + str(pos.row + 1)


def in_bounds(pos: BoardPoint) -> bool:
    return 0 <= pos.column <= 7 and 0 <= pos.row <= 7


def is_lightsquare(pos: BoardPoint) -> bool:
    return pos.column % 2 != pos.row % 2
