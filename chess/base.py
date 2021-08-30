import collections

FENSYMBOLS = {
    "k": "King",
    "q": "Queen",
    "r": "Rook",
    "b": "Bishop",
    "n": "Knight",
    "p": "Pawn",
}
STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BoardPoint = collections.namedtuple("BoardPoint", ("column", "row"), module="chess")

class error(ValueError):
    pass

def decode_pos(pos: str) -> BoardPoint:
    return BoardPoint(ord(pos[0]) - 97, int(pos[1]) - 1)


def encode_pos(pos: BoardPoint) -> str:
    return chr(pos.column + 97) + str(pos.row + 1)


def in_bounds(pos: BoardPoint) -> bool:
    return 0 <= pos.column <= 7 and 0 <= pos.row <= 7