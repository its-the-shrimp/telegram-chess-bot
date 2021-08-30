from .matches import BaseMatch, GroupMatch, PMMatch, AIMatch, from_dict
from .board import Move, BoardInfo
from .pieces import BasePiece, Pawn, Knight, Bishop, Rook, Queen, King
from .base import encode_pos, decode_pos, BoardPoint, STARTPOS

MODES = [{"text": "Против бота", "code": "AI"}, {"text": "Онлайн", "code": "QUICK"}]
