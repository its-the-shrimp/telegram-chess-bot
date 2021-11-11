from .base import *

from . import media, analysis
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
    MoveEval,
    GameState,
)
from .matches import (
    BaseMatch,
    GroupMatch,
    PMMatch,
    AIMatch,
    from_bytes,
    get_pgn_file,
)
from .parsers import PGNParser, CGNParser, get_moves


def init(is_debug: bool):
    BaseMatch.ENGINE_FILENAME = "./stockfish" if is_debug else "./stockfish_14_x64"
    BaseMatch.db = get_database()


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
        "values": {
            "low-diff": None,
            "mid-diff": None,
            "high-diff": None,
            "max-diff": None,
        },
        "condition": lambda obj: obj["mode"] == "vsbot",
    },
}
KEYBOARD_COMMANDS: dict[str, Callable[[Update, CallbackContext, list[Union[str, int]]], None]] = {"DOWNLOAD": get_pgn_file}
INVITE_IMAGE = "https://raw.githubusercontent.com/schvv31n/telegram-chess-bot/master/images/static/inline-thumb.jpg"
