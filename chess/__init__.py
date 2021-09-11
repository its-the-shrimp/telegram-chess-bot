from . import media, analysis
from .utils import encode_pos, decode_pos, BoardPoint, STARTPOS
from .core import Move, BoardInfo, get_moves, get_pgn_moveseq, BasePiece, Pawn, Knight, Bishop, Rook, Queen, King
from .matches import BaseMatch, GroupMatch, PMMatch, AIMatch, from_dict

MODES = [{"text": "Против бота", "code": "AI"}, {"text": "Онлайн", "code": "QUICK"}]
