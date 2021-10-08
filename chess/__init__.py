from . import media, analysis
from .utils import encode_pos, decode_pos, BoardPoint, STARTPOS, langtable, InlineMessageAdapter, create_match_id
from .core import (
    Move,
    BoardInfo,
    get_moves,
    get_pgn_moveseq,
    decode_pgn_moveseq,
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
    from_dict,
    get_pgn_file,
)


def init(is_debug: bool, conn):
    BaseMatch.ENGINE_FILENAME = "./stockfish" if is_debug else "./stockfish_14_x64"
    BaseMatch.db = conn


OPTIONS = {
    "ruleset": {"options": {"std-chess": {}}, "conditions": {}},
    "mode": {
        "options": {
            "online": {},
            "vsbot": {"ruleset": lambda ruleset: ruleset != "fog-of-war"},
            "invite": {},
        },
        "conditions": {},
    },
    "timectrl": {
        "options": {"classic": {}, "rapid": {}, "blitz": {}},
        "conditions": {"mode": lambda mode: mode != "vsbot"},
    },
    "difficulty": {
        "options": {"low-diff": {}, "mid-diff": {}, "high-diff": {}, "max-diff": {}},
        "conditions": {"mode": lambda mode: mode == "vsbot"},
    },
}
KEYBOARD_BUTTONS = {"DOWNLOAD": get_pgn_file}
INVITE_IMAGE = "https://raw.githubusercontent.com/schvv31n/telegram-chess-bot/master/images/chess/board.png"
INVITE_THUMBNAIL = "https://raw.githubusercontent.com/schvv31n/telegram-chess-bot/master/images/chess/inline-thumb.png"
