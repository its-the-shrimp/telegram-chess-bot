from . import media, analysis
from .base import (
    format_callback_data,
    parse_callback_data,
    langtable,
    InlineMessageAdapter,
    create_match_id,
    get_file_url,
    get_tempfile_url,
)
from .utils import BoardPoint, STARTPOS
from .core import (
    Move,
    BoardInfo,
    BasePiece,
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
)
from .matches import (
    BaseMatch,
    GroupMatch,
    PMMatch,
    AIMatch,
    from_bytes,
    get_pgn_file,
)
from .parsers import PGNParser, CGNParser


def init(is_debug: bool, conn):
    BaseMatch.ENGINE_FILENAME = "./stockfish" if is_debug else "./stockfish_14_x64"
    BaseMatch.db = conn


OPTIONS = {
    "ruleset": {"values": {"std-chess": None}},
    "mode": {
        "values": {
            "online": None,
            "vsbot": lambda obj: obj["ruleset"] != "fog-of-war",
            "invite": None,
        },
    },
    "timectrl": {
        "values": {"classic": None, "rapid": None, "blitz": None},
        "condition": lambda obj: obj["mode"] != "vsbot",
    },
    "difficulty": {
        "values": {"low-diff": None, "mid-diff": None, "high-diff": None, "max-diff": None},
        "condition": lambda obj: obj["mode"] == "vsbot",
    }
}
KEYBOARD_BUTTONS = {"DOWNLOAD": get_pgn_file}
INVITE_IMAGE = "https://avatars.githubusercontent.com/u/73731786?s=400&u=8e7a61eb0beaef03fbb151a70861097ae3c90fdf&v=4"
