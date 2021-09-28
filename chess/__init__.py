from . import media, analysis
from .utils import encode_pos, decode_pos, BoardPoint, STARTPOS, langtable
from .core import Move, BoardInfo, get_moves, get_pgn_moveseq, BasePiece, Pawn, Knight, Bishop, Rook, Queen, King
from .matches import BaseMatch, GroupMatch, PMMatch, AIMatch, from_dict
OPTIONS = {
    "ruleset": {
        "options": ["std-chess"],
        "conditions": {}
    },
    "mode": {
        "options": ["online", "vsbot", "invite"],
        "conditions": {}
    },
    "timectrl": {
        "options": ["classic", "rapid", "blitz"],
        "conditions": {"mode": lambda mode: mode != "vsbot"}
    },
    "difficulty": {
        "options": ["low-diff", "mid-diff", "high-diff", "max-diff"],
        "conditions": {"mode": lambda mode: mode == "vsbot"}
    }
}