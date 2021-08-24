import itertools
import random
import subprocess
import numpy
import os
from telegram import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    InputMediaPhoto,
    InputMediaVideo,
    User,
    Message,
    Chat,
    Bot,
)
import cv2
import cv_utils
import logging
import time
import collections
from typing import Any, Union, List, Type, Optional

IDSAMPLE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-+"
MODES = [{"text": "Против бота", "code": "AI"}, {"text": "Онлайн", "code": "QUICK"}]
MOVETYPE_MARKERS = {"normal": "", "killing": "❌", "castling": "🔀", "promotion": "⏫"}
FENSYMBOLS = {
    "k": "King",
    "q": "Queen",
    "r": "Rook",
    "b": "Bishop",
    "n": "Knight",
    "p": "Pawn",
}
PGNSYMBOLS = {
    "emoji": {
        "Pawn": "",
        "Rook": "♜",
        "Bishop": "♝",
        "Knight": "♞",
        "Queen": "♛",
        "King": "♚",
    },
    "en": {
        "Pawn": "",
        "Rook": "R",
        "Bishop": "B",
        "Knight": "N",
        "Queen": "Q",
        "King": "K",
    },
}
IMAGES = {}
for name in ["Pawn", "King", "Bishop", "Rook", "Queen", "Knight"]:
    IMAGES[name] = [
        cv2.imread(f"images/chess/{color}_{name.lower()}.png", cv2.IMREAD_UNCHANGED)
        for color in ["black", "white"]
    ]
STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MOVETYPE_COLORS = {
    "normal": "#00cc36",
    "killing": "cc0000",
    "castling": "#3ba7ff",
    "promotion": "#3ba7ff",
    "killing-promotion": "#3ba7ff",
}
BOARD_IMG = cv2.imread("images/chess/board.png", cv2.IMREAD_UNCHANGED)

POINTER_IMG = numpy.zeros((50, 50, 4), dtype=numpy.uint8)
width = POINTER_IMG.shape[0] // 2
for row in range(POINTER_IMG.shape[0]):
    for column in range(POINTER_IMG.shape[1]):
        if abs(row - width) + abs(column - width) > round(width * 1.5):
            POINTER_IMG[row][column] = cv_utils.from_hex(MOVETYPE_COLORS["normal"]) + [
                255
            ]
        elif abs(row - width) + abs(column - width) == round(width * 1.5):
            POINTER_IMG[row][column] = cv_utils.from_hex(MOVETYPE_COLORS["normal"]) + [
                127
            ]
INCOMING_POINTER_IMG = numpy.zeros((50, 50, 4), dtype=numpy.uint8)
for row in range(INCOMING_POINTER_IMG.shape[0]):
    for column in range(INCOMING_POINTER_IMG.shape[1]):
        if abs(row - width) + abs(column - width) < round(width * 0.5):
            INCOMING_POINTER_IMG[row][column] = cv_utils.from_hex(
                MOVETYPE_COLORS["normal"]
            ) + [255]
        elif abs(row - width) + abs(column - width) == round(width * 0.5):
            INCOMING_POINTER_IMG[row][column] = cv_utils.from_hex(
                MOVETYPE_COLORS["normal"]
            ) + [127]
del width, column, row

BoardPoint = collections.namedtuple("BoardPoint", ("column", "row"), module="chess")
JSON = dict[str, Union[str, dict]]


def _group_items(
    obj: list[Any], n: int, head_item: bool = False
) -> list[list[Any]]:
    res = []
    for index in range(len(obj)):
        index -= int(head_item)
        if index == -1:
            res.append([obj[0]])
        elif index // n == index / n:
            res.append([obj[index + int(head_item)]])
        else:
            res[-1].append(obj[index + int(head_item)])

    return res


def decode_fen(fen: str) -> dict[tuple, str]:
    res = {}
    fen = fen.split("/")
    for line in range(8):
        offset = 0
        for column in range(8):
            if column + offset > 7:
                break
            char = fen[line][column]
            if char.isdigit():
                offset += int(char) - 1
            else:
                res[(column + offset, 7 - line)] = char

    return res


def decode_pos(pos: str) -> BoardPoint:
    return BoardPoint(ord(pos[0]) - 97, int(pos[1]) - 1)


def encode_pos(pos: BoardPoint) -> str:
    return chr(pos.column + 97) + str(pos.row + 1)


def in_bounds(pos: BoardPoint) -> bool:
    return 0 <= pos.column <= 7 and 0 <= pos.row <= 7


def from_dict(obj: dict[str, Any], match_id: str, bot: Bot) -> "BaseMatch":
    cls = eval(obj["type"] + "Match")
    return cls.from_dict(obj, match_id, bot)


class Move:
    @classmethod
    def from_pgn(cls, move: str, board_obj: "BoardInfo", language_code: str = "en"):
        symbols = {v: k for k, v in PGNSYMBOLS[language_code].items()}
        row = 0 if board_obj.is_white_turn else 7
        move = move.strip("#+")

        if move == "O-O":
            return cls(
                board_obj,
                BoardPoint(4, row),
                BoardPoint(6, row),
                King,
                rook_src=BoardPoint(7, row),
                rook_dst=BoardPoint(5, row),
            )
        elif move == "O-O-O":
            return cls(
                board_obj,
                BoardPoint(4, row),
                BoardPoint(2, row),
                King,
                rook_src=BoardPoint(0, row),
                rook_dst=BoardPoint(3, row),
            )
        else:
            if move[0] in "abcdefghx":
                piece_cls = Pawn
            else:
                piece_cls = eval(symbols[move[0]])
                move = move[1:]

            if "x" in move:
                killing = True
                hint, move = move.split("x")
            else:
                killing = False
                if len(move) == 3:
                    hint = move[0]
                    move = move[1:]
                else:
                    hint = ""

            if "=" in move:
                move, promoted_to = move.split("=")
                promoted_to = eval(symbols[promoted_to])
            else:
                promoted_to = None

            if hint and hint in "abcdefgh":
                row_hint = None
                column_hint = ord(hint) - 97
            elif hint.isdigit():
                row_hint = int(hint) - 1
                column_hint = None
            else:
                row_hint, column_hint = (None, None)

            target = piece_cls(decode_pos(move), board_obj, board_obj.is_white_turn)
            for src in target.get_all_moves(target.pos, target.is_white, is_killing=killing):
                piece = board_obj[src]
                if all(
                    [
                        column_hint in [src[0], None],
                        row_hint in [src[1], None],
                        (target.is_white == piece.is_white) if piece else False,
                        type(piece) == piece_cls,
                    ]
                ):
                    if (
                        piece_cls != Pawn
                        or bool(abs(src[0] - target.pos[0])) == killing
                    ):
                        break

        return cls(
            board_obj,
            src,
            target.pos,
            piece_cls,
            killed=type(board_obj[target.pos]),
            new_piece=promoted_to,
        )

    def __init__(
        self,
        board_obj: "BoardInfo",
        src: BoardPoint,
        dst: BoardPoint,
        piece: Type["BasePiece"],
        killed: Optional[Type["BasePiece"]] = type(None),
        rook_src: BoardPoint = None,
        rook_dst: BoardPoint = None,
        new_piece: type = None,
    ):
        assert board_obj[src] is not None
        self.src = src
        self.dst = dst
        self.piece = piece
        self.killed = killed
        self.new_piece = new_piece
        self.rook_src = rook_src
        self.rook_dst = rook_dst
        self.board = board_obj

    def __repr__(self):
        return f"Move({self.pgn})"

    @property
    def is_killing(self) -> bool:
        return self.killed != type(None)

    @property
    def is_castling(self) -> bool:
        return self.piece == King and self.rook_src

    @property
    def is_promotion(self) -> bool:
        return self.piece == Pawn and bool(self.new_piece)

    @property
    def enpassant_pos(self) -> list[Optional[BoardPoint]]:
        if self.piece == Pawn and abs(self.src[1] - self.dst[1]) == 2:
            return [self.dst, BoardPoint(self.dst[0], (self.src[1] + self.dst[1]) // 2)]
        return [None, None]

    @property
    def type(self) -> str:
        if self.is_killing and self.is_promotion:
            return "killing-promotion"
        elif self.is_promotion:
            return "promotion"
        elif self.is_killing:
            return "killing"
        elif self.is_castling:
            return "castling"
        return "normal"

    @property
    def opponent_state(self) -> str:
        king = self.board[King, not self.board.is_white_turn][0]
        if king.in_checkmate():
            return "checkmate"
        elif king.in_check():
            return "check"
        return "normal"

    @property
    def pgn(self, language_code: str = "en") -> str:
        if self.is_castling:
            return "-".join(["O"] * abs(self.rook_src[0] - self.rook_dst[0]))
        move = ("x" if self.is_killing else "") + encode_pos(self.dst)
        positions = []
        for pos in self.piece.get_all_moves(
            self.dst, self.board.is_white_turn, is_killing=self.is_killing
        ):
            piece = self.board[pos]
            if type(piece) == self.piece and piece.is_white == self.board.is_white_turn:
                positions.append(piece.pos)

        if len(positions) > 1:
            if len({i[0] for i in positions}) == len([i[0] for i in positions]):
                move = encode_pos(self.src)[0] + move
            else:
                move = encode_pos(self.src)[1] + move

        move = PGNSYMBOLS[language_code][self.piece.__name__] + move
        if self.is_promotion:
            move += "=" + PGNSYMBOLS[language_code][self.new_piece.__name__]
        move += {"normal": "", "check": "+", "checkmate": "#"}[self.opponent_state]

        return move

    def copy(self, **params) -> "Move":
        defaults = {
            "board_obj": self.board,
            "src": self.src,
            "dst": self.dst,
            "piece": self.piece,
            "killed": self.killed,
            "rook_src": self.rook_src,
            "rook_dst": self.rook_dst,
            "new_piece": self.new_piece,
        }

        return type(self)(**(defaults | params))


class BoardInfo:
    @classmethod
    def from_fen(cls, fen: str, **kwargs) -> None:
        res = []
        (
            board,
            is_white_turn,
            castlings,
            enpassant_pos,
            empty_halfturns,
            turn,
        ) = [int(i) if i.isdigit() else i for i in fen.split(" ")]

        is_white_turn = is_white_turn == "w"

        if enpassant_pos == "-":
            enpassant_pos = [None, None]
        else:
            enpassant_pos = decode_pos(enpassant_pos)
            enpassant_pos = [
                [enpassant_pos[0], int((enpassant_pos[1] + 4.5) // 2)],
                enpassant_pos,
            ]

        for (column, line), char in decode_fen(board).items():
            new = eval(FENSYMBOLS[char.lower()])(
                BoardPoint(column, line), None, char.isupper()
            )
            K = "K" if new.is_white else "k"
            Q = "Q" if new.is_white else "q"
            if type(new) == King and K not in castlings and Q not in castlings:
                new.moved = True
            elif (
                type(new) == Rook
                and K not in castlings
                and Q in castlings
                and new.pos != [0, 0 if new.is_white else 7]
            ):
                new.moved = True
            elif (
                type(new) == Rook
                and K in castlings
                and Q not in castlings
                and new.pos != [7, 0 if new.is_white else 7]
            ):
                new.moved = True
            res.append(new)

        return cls(
            res,
            is_white_turn=is_white_turn,
            enpassant_pos=enpassant_pos,
            empty_halfturns=empty_halfturns,
            turn=turn,
            **kwargs,
        )

    def __new__(
        cls,
        board: list["BasePiece"],
        is_white_turn: bool = True,
        enpassant_pos: list[BoardPoint] = [None, None],
        empty_halfturns: int = 0,
        turn: int = 1,
        **kwargs,
    ):
        self = object.__new__(cls)

        self.board = board
        self.is_white_turn = is_white_turn
        self.enpassant_pos = enpassant_pos
        self.empty_halfturns = empty_halfturns
        self.turn = turn

        for piece in board:
            piece.board = self

        return self

    def __repr__(self):
        return f"BoardInfo({self.fen})"

    def __eq__(self, other: "BoardInfo"):
        for attr in ["__class__", "board", "castlings", "enpassant_pos"]:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __sub__(self, other: "BoardInfo") -> Move:
        src_pieces = []
        for piece in other.board:
            if piece not in self.board:
                src_pieces.append(piece)

        dst_pieces = []
        for piece in self.board:
            if piece not in other.board:
                dst_pieces.append(piece)

        if len(dst_pieces) == 2:
            _, src_king = sorted(src_pieces, key=lambda x: x.value)
            _, dst_king = sorted(dst_pieces, key=lambda x: x.value)
            return src_king.create_move(dst_king.pos)

        src_piece = sorted(src_pieces, key=lambda x: int(x.is_white == other.is_white_turn))[-1]
        return src_piece.create_move(
            dst_pieces[0].pos,
            new_piece=type(dst_pieces[0]) if type(dst_pieces[0])!=type(src_piece) else None
        )        

    def __add__(self, move: Move) -> "BoardInfo":
        new = self.copy(
            turn=self.turn + (1 if not self.is_white_turn else 0),
            is_white_turn=not self.is_white_turn,
            empty_halfturns=0
            if move.is_killing or move.piece == Pawn
            else self.empty_halfturns + 1,
            enpassant_pos=move.enpassant_pos,
        )
        piece = new[move.src]

        if move.is_killing:
            del new[move.dst]
        if move.is_promotion:
            del new[move.src]
            new.board.append(move.new_piece(piece.pos, self, self.is_white_turn))
        if move.is_castling:
            new[move.rook_src] = move.rook_dst

        new[move.src] = move.dst

        return new

    def __getitem__(self, *keys):
        keys = keys[0]
        if type(keys) == int:
            return self.get_by_pos(keys, None)
        elif type(keys) == type:
            return self.get_by_type(keys, None)
        elif issubclass(type(keys), tuple):
            if type(keys[0]) == int:
                return self.get_by_pos(*keys)
            elif type(keys[0]) == type or type(keys[1]) == bool:
                return self.get_by_type(*keys)
            elif keys == (None, None):
                return self.board
        raise TypeError(f"Unknown argument: {keys}")

    def __setitem__(self, pos: BoardPoint, new_pos: BoardPoint):
        self[pos]._move(new_pos)

    def __delitem__(self, pos):
        for index in range(len(self.board)):
            if self.board[index].pos == pos:
                del self.board[index]
                break

    @property
    def whites(self) -> list["BasePiece"]:
        return list(filter(lambda x: x.is_white, self.board))

    @property
    def blacks(self) -> list["BasePiece"]:
        return list(filter(lambda x: not x.is_white, self.board))

    @property
    def castlings(self) -> str:
        res = ""
        white_king = self[King, True][0]
        black_king = self[King, False][0]
        if not white_king.moved:
            white_king_moves = [i.dst for i in white_king.get_moves(for_fen=True)]
            if BoardPoint(6, 0) in white_king_moves:
                res += "K"
            if BoardPoint(2, 0) in white_king_moves:
                res += "Q"
        if not black_king.moved:
            black_king_moves = [i.dst for i in black_king.get_moves(for_fen=True)]
            if BoardPoint(6, 7) in black_king_moves:
                res += "k"
            if BoardPoint(2, 7) in black_king_moves:
                res += "q"
        return res if res else "-"

    @property
    def fen(self):
        board = []
        for row in range(7, -1, -1):
            board.append("")
            for column in range(0, 8):
                piece = self[column, row]
                if piece:
                    board[-1] += piece.fen_symbol
                elif board[-1] and board[-1][-1].isdigit():
                    board[-1] = board[-1][:-1] + str(int(board[-1][-1]) + 1)
                else:
                    board[-1] += "1"

        return " ".join(
            (
                "/".join(board),
                "w" if self.is_white_turn else "b",
                self.castlings,
                encode_pos(self.enpassant_pos[1]) if self.enpassant_pos[0] else "-",
                str(self.empty_halfturns),
                str(self.turn),
            )
        )

    def copy(self, **new_params):
        params = {
            "is_white_turn": self.is_white_turn,
            "enpassant_pos": self.enpassant_pos,
            "empty_halfturns": self.empty_halfturns,
            "turn": self.turn,
        }
        return type(self)([piece.copy() for piece in self.board], **(params | new_params))

    def get_by_pos(self, column, row):
        res = []
        for piece in self.board:
            if (piece.pos[0] == column or column is None) and (
                piece.pos[1] == row or row is None
            ):
                res.append(piece)

        if column != None and row != None:
            return res[0] if res else None
        else:
            return res

    def get_by_type(self, piece_type: type, is_white: bool):
        res = []
        for piece in self.board:
            if (type(piece) == piece_type or piece_type is None) and (
                piece.is_white == is_white or is_white is None
            ):
                res.append(piece)

        return res

    def is_legal(self, move: Move):
        test_obj = self + move
        return not test_obj[King, self.is_white_turn][0].in_check()


class BasePiece:
    name: str
    fen_symbol: list[str]
    pgn_symbol: dict[str, str]
    move_diffs: Union[tuple[list[int]], tuple[tuple[list[int]]]]
    value: int

    def __init__(self, pos: BoardPoint, board: BoardInfo, is_white: bool):
        self.image = IMAGES[type(self).__name__][int(is_white)]
        self.pos = pos
        self.is_white = is_white
        self.board = board
        self.fen_symbol = self.fen_symbol[int(is_white)]
        self.moved = False

    def __str__(self):
        return f"{self.name} на {encode_pos(self.pos)}"

    def __repr__(self):
        return f"<{type(self).__name__}({self.pos}, {self.is_white})>"

    def __eq__(self, other: "BasePiece"):
        return (
            type(self) == type(other)
            and self.pos == other.pos
            and self.is_white == other.is_white
        )

    @property
    def allied_pieces(self) -> list["BasePiece"]:
        return self.board.whites if self.is_white else self.board.blacks

    @property
    def enemy_pieces(self) -> list["BasePiece"]:
        return self.board.blacks if self.is_white else self.board.whites

    def _move(self, new_pos: BoardPoint):
        self.pos = new_pos
        self.moved = True

    def get_moves(self, **kwargs) -> list[Move]:
        return []

    @classmethod
    def get_all_moves(
        cls, pos: BoardPoint, is_white: bool, diffs: List = tuple(), **kwargs
    ) -> list[BoardPoint]:
        res = []
        for diff in diffs if diffs else cls.move_diffs:
            res.append(BoardPoint(*[i + d for i, d in zip(pos, diff)]))

        return list(filter(in_bounds, res))

    def remove(self) -> None:
        del self.board[self.pos]

    def create_move(self, dst: BoardPoint, new_piece: Type["BasePiece"] = None) -> Move:
        piece = type(self)
        kwargs = {"killed": type(self.board[dst])}
        if piece == King:
            if self.pos.column == 4 and dst.column == 6:  # kingside castling
                kwargs["rook_src"] = BoardPoint(7, self.pos.row)
                kwargs["rook_dst"] = BoardPoint(5, dst.row)
            elif self.pos.column == 4 and dst.column == 2:  # queenside castling
                kwargs["rook_src"] = BoardPoint(0, self.pos.row)
                kwargs["rook_dst"] = BoardPoint(3, dst.row)

        return Move(self.board, self.pos, dst, piece, new_piece=new_piece, **kwargs)

    def copy(self) -> "BasePiece":
        new = type(self)(self.pos, self.board, self.is_white)
        new.moved = self.moved
        return new


class Pawn(BasePiece):
    name = "Пешка"
    fen_symbol = ["p", "P"]
    pgn_symbol = {"emoji": "", "ru": "", "en": ""}
    move_diffs = ([0, -1], [0, -2], [1, -1], [-1, -1])
    value = 1

    @classmethod
    def get_all_moves(
        cls: type, pos: BoardPoint, is_white: bool, is_killing: bool = False, **kwargs
    ) -> list[BoardPoint]:
        diffs = cls.move_diffs
        if not is_killing:
            diffs = diffs[:2]
        if not is_white:
            diffs = [[i[0], i[1] * -1] for i in diffs]

        res = []
        for diff in diffs:
            res.append(BoardPoint(*[i + d for i, d in zip(pos, diff)]))

        return list(filter(in_bounds, res))

    def get_moves(self, **kwargs) -> list[Move]:
        positions = []
        direction = 1 if self.is_white else -1
        if BoardPoint(self.pos.column, self.pos.row + direction) not in [
            i.pos for i in self.enemy_pieces
        ]:
            positions.append(BoardPoint(self.pos.column, self.pos.row + direction))
            if self.pos.row == (1 if self.is_white else 6) and BoardPoint(
                self.pos.column, self.pos.row + direction * 2
            ) not in [i.pos for i in self.enemy_pieces]:
                positions.append(
                    BoardPoint(self.pos.column, self.pos.row + direction * 2)
                )

        if BoardPoint(self.pos.column + 1, self.pos.row + direction) in [
            i.pos for i in self.enemy_pieces
        ] + [self.board.enpassant_pos[1]]:
            positions.append(BoardPoint(self.pos.column + 1, self.pos.row + direction))

        if BoardPoint(self.pos.column - 1, self.pos.row + direction) in [
            i.pos for i in self.enemy_pieces
        ] + [self.board.enpassant_pos[1]]:
            positions.append(BoardPoint(self.pos.column - 1, self.pos.row + direction))

        moves = []
        for move in positions:
            if in_bounds(move) and move not in [i.pos for i in self.allied_pieces]:
                moves.append(self.create_move(move))

        return moves


class Knight(BasePiece):
    name = "Конь"
    fen_symbol = ["n", "N"]
    move_diffs = [
        [2, -1],
        [2, 1],
        [1, 2],
        [1, -2],
        [-1, 2],
        [-1, -2],
        [-2, 1],
        [-2, -1],
    ]
    value = 3

    def get_moves(self, **kwargs) -> list[Move]:
        moves = []
        for move in self.get_all_moves(self.pos, self.is_white):
            if in_bounds(move) and move not in [i.pos for i in self.allied_pieces]:
                moves.append(self.create_move(move))

        return moves


class Rook(BasePiece):
    name = "Ладья"
    fen_symbol = ["r", "R"]
    pgn = {"emoji": "♜", "ru": "Л", "en": "R"}
    move_diffs = (
        tuple(zip(range(1, 8), [0] * 7)),
        tuple(zip(range(-1, -8, -1), [0] * 7)),
        tuple(zip([0] * 7, range(1, 8))),
        tuple(zip([0] * 7, range(-1, -8, -1))),
    )
    value = 5

    @classmethod
    def get_all_moves(cls, pos, is_white, flat=True, **kwargs) -> list[BoardPoint]:
        res = [
            BasePiece.get_all_moves(pos, is_white, diffs=seq) for seq in cls.move_diffs
        ]
        if flat:
            return list(itertools.chain(*res))
        else:
            return res

    def get_moves(self, **kwargs) -> list[Move]:
        moves = []
        for move_seq in self.get_all_moves(self.pos, self.is_white, flat=False):
            for move in move_seq:
                if move in [i.pos for i in self.allied_pieces] or not in_bounds(move):
                    break
                elif move in [i.pos for i in self.enemy_pieces] + [
                    self.board.enpassant_pos[1]
                ]:
                    moves.append(self.create_move(move))
                    break
                else:
                    moves.append(self.create_move(move))

        return moves


class Bishop(Rook):
    name = "Слон"
    fen_symbol = ["b", "B"]
    pgn_symbol = {"emoji": "♝", "ru": "С", "en": "B"}
    move_diffs = (
        tuple(zip(range(1, 8), range(1, 8))),
        tuple(zip(range(-1, -8, -1), range(-1, -8, -1))),
        tuple(zip(range(-1, -8, -1), range(1, 8))),
        tuple(zip(range(1, 8), range(-1, -8, -1))),
    )
    value = 3


class Queen(Rook):
    name = "Ферзь"
    fen_symbol = ["q", "Q"]
    pgn_symbol = {"emoji": "♛", "ru": "Ф", "en": "Q"}
    value = 8

    @classmethod
    def get_all_moves(self, pos, is_white, flat=True, **kwargs):
        return Bishop.get_all_moves(pos, is_white, flat=flat) + super().get_all_moves(
            pos, is_white, flat=flat
        )


class King(BasePiece):
    name = "Король"
    fen_symbol = ["k", "K"]
    pgn_symbol = {"emoji": "♚", "ru": "Кр", "en": "K"}
    move_diffs = tuple(
        [i for i in itertools.product([1, 0, -1], [1, 0, -1]) if i != (0, 0)]
    )
    value = 99

    def get_moves(self, for_fen: bool = False, castling: bool = True) -> list[Move]:
        moves = []
        for move in self.get_all_moves(self.pos, self.is_white):
            if in_bounds(move) and move not in [i.pos for i in self.allied_pieces]:
                moves.append(move)

        if castling and not self.moved and not self.in_check():
            Y = 0 if self.is_white else 7
            a_rook = self.board[0, Y]
            h_rook = self.board[7, Y]
            if (
                all([not self.board[x, Y] or for_fen for x in [1, 2, 3]])
                and a_rook
                and not a_rook.moved
            ):
                moves.append(BoardPoint(2, Y))
            if (
                all([not self.board[x, Y] or for_fen for x in [5, 6]])
                and h_rook
                and not h_rook.moved
            ):
                moves.append(BoardPoint(6, Y))

        return [self.create_move(move) for move in moves]

    def in_checkmate(self):
        checks = []
        for piece in self.allied_pieces:
            for move in piece.get_moves():
                checks.append(move)

        return (
            next(filter(self.board.is_legal, checks), None) is None and self.in_check()
        )

    def in_check(self):
        enemy_moves = itertools.chain(
            *[i.get_moves(castling=False) for i in self.enemy_pieces]
        )

        return self.pos in [i.dst for i in enemy_moves]


class BaseMatch:
    WRONG_PERSON_MSG = "Сейчас не ваш ход!"
    db = None

    def __init__(self, bot=None, id=None):
        self.init_msg_text: Optional[str] = None
        self.last_move: Optional[Move] = None
        self.bot: Bot = bot
        self.states: list[BoardInfo] = [BoardInfo.from_fen(STARTPOS)]
        self.finished: bool = False
        self.id: str = id if id else "".join(random.choices(IDSAMPLE, k=8))
        self.image_filename: str = f"chess-{self.id}.jpg"
        self.video_filename: str = f"chess-{self.id}.mp4"
        self.game_filename: str = f"telegram-chess-bot-{self.id}.pgn"

    def _keyboard(
        self,
        seq: list[dict[str, Union[str, BoardPoint]]],
        expected_uid: int,
        head_item: bool = False,
    ) -> Optional[InlineKeyboardMarkup]:
        res = []
        for button in seq:
            data = []
            for argument in button["data"]:
                if type(argument) == BoardPoint:
                    data.append(encode_pos(argument))
                elif argument == None:
                    data.append("")
                else:
                    data.append(str(argument))
            res.append(
                InlineKeyboardButton(
                    text=button["text"],
                    callback_data=f"{expected_uid if expected_uid else ''}\n{self.id}\n{'#'.join(data)}",
                )
            )

        if res:
            return InlineKeyboardMarkup(_group_items(res, 2, head_item=head_item))
        else:
            return None

    def get_moves(self) -> list[Move]:
        res = [None]
        for index in range(1, len(self.states)):
            res.append(self.states[index] - self.states[index - 1])

        return res

    def to_dict(self) -> JSON:
        return {"type": "Base", "states": [board.fen for board in self.states]}

    @property
    def board(self) -> BoardInfo:
        return self.states[-1]

    @property
    def pieces(self) -> tuple[list[BasePiece], list[BasePiece]]:
        return (self.board.whites, self.board.blacks) if self.board.is_white_turn else (self.board.blacks, self.board.whites)

    def get_state(self) -> Optional[str]:
        cur_king = self.board[King, self.board.is_white_turn][0]
        if cur_king.in_checkmate():
            return "checkmate"
        if cur_king.in_check():
            return "check"

        if self.board.empty_halfturns >= 50:
            return "50-move-draw"
        cur_side_moves = [piece.get_moves() for piece in cur_king.allied_pieces]
        if len(self.states) >= 8:
            for move in itertools.chain(
                *(
                    [piece.get_moves() for piece in cur_king.enemy_pieces]
                    + cur_side_moves
                )
            ):
                test_board = self.board + move
                if test_board == self.states[-8]:
                    return "3fold-repetition-draw"
        if next(itertools.chain(*cur_side_moves), None) is None:
            return "stalemate-draw"

        return "normal"

    def init_turn(self, move: Move = None) -> None:
        if move:
            self.states.append(self.states[-1] + move)
        self.last_move = move

    def visualise_board(
        self,
        board: BoardInfo=None,
        selected: BoardPoint=None,
        possible_moves: list[Move] = [],
        prev_move: Move=-1,
        return_bytes=True,
    ) -> Union[bytes, numpy.ndarray]:
        board_img = BOARD_IMG.copy()
        board = board if board else self.board
        prev_move = prev_move if prev_move != -1 else self.last_move
        logging.debug(repr(prev_move))
        for piece in board.board:
            if piece.pos == selected:
                cv_utils.fill(
                    cv_utils.from_hex("#00cc36"),
                    board_img,
                    topleft=cv_utils.image_pos(piece.pos, piece.image.shape[:2]),
                    mask=piece.image,
                )
            else:
                cv_utils.paste(
                    piece.image,
                    board_img,
                    cv_utils.image_pos(piece.pos, piece.image.shape[:2]),
                )

        for new_move in possible_moves:
            cv_utils.fill(
                cv_utils.from_hex(MOVETYPE_COLORS[new_move.type]),
                board_img,
                topleft=cv_utils.image_pos(new_move.dst, POINTER_IMG.shape[:2]),
                mask=POINTER_IMG,
            )

        for king in board[King]:
            if king.in_check():
                cv_utils.fill(
                    cv_utils.from_hex(MOVETYPE_COLORS["killing"]),
                    board_img,
                    topleft=cv_utils.image_pos(
                        king.pos, INCOMING_POINTER_IMG.shape[:2]
                    ),
                    mask=INCOMING_POINTER_IMG,
                )

        if prev_move:
            for move in [prev_move.src, prev_move.dst]:
                cv_utils.fill(
                    cv_utils.from_hex(MOVETYPE_COLORS["normal"]),
                    board_img,
                    topleft=cv_utils.image_pos(move, INCOMING_POINTER_IMG.shape[:2]),
                    mask=INCOMING_POINTER_IMG,
                )

        return cv2.imencode(".jpg", board_img)[1].tobytes() if return_bytes else board_img

    def get_video(self) -> tuple[bytes, bytes]:
        path = os.path.join("images", "temp", self.video_filename)
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, BOARD_IMG.shape[1::-1]
        )

        for board, move in zip(self.states, self.get_moves()):
            img_array = self.visualise_board(board=board, prev_move=move, return_bytes=False)
            for i in range(15):
                writer.write(img_array)
        for i in range(15):
            writer.write(img_array)

        thumbnail = cv2.resize(img_array, None, fx=0.5, fy=0.5)
        writer.release()
        video_data = open(path, "rb").read()
        os.remove(path)
        return video_data, cv2.imencode(".jpg", thumbnail)[1].tobytes()


class GroupMatch(BaseMatch):
    def __init__(self, player1: User, player2: User, match_chat: int, **kwargs):
        self.player1: User = player1
        self.player2: User = player2
        self.chat_id: int = match_chat
        self.msg: Message = None
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, obj: JSON, match_id: int, bot: Bot) -> "GroupMatch":
        logging.debug(f"Constructing {cls.__name__} object:", obj)
        player1 = User.de_json(obj["player1"] | {"is_bot": False}, bot)
        player2 = User.de_json(obj["player2"] | {"is_bot": False}, bot)
        new = cls(
            player1,
            player2,
            obj["chat_id"],
            bot=bot,
            id=match_id,
        )
        new.states = [BoardInfo.from_fen(fen) for fen in obj["states"]]
        new.init_msg_text = obj["msg_text"]
        new.msg = Message(
            obj["msg_id"],
            time.monotonic,
            Chat(obj["chat_id"], "group", bot=bot),
            bot=bot,
            caption=obj["msg_text"],
        )
        new.turn += 1
        new.empty_halfturns += 1
        new.is_white_turn = not new.is_white_turn

        return new

    @property
    def players(self) -> tuple[User, User]:
        return (
            (self.player1, self.player2)
            if self.board.is_white_turn
            else (self.player2, self.player1)
        )

    def to_dict(self) -> JSON:
        res = super().to_dict()
        res.update(
            {
                "type": "Group",
                "chat_id": self.chat_id,
                "msg_id": self.msg.message_id,
                "msg_text": self.init_msg_text,
                "player1": {
                    k: v
                    for k, v in self.player1.to_dict().items()
                    if k in ["username", "id", "first_name", "last_name"]
                },
                "player2": {
                    k: v
                    for k, v in self.player2.to_dict().items()
                    if k in ["username", "id", "first_name", "last_name"]
                },
            }
        )
        return res

    def init_turn(self, move: Move = None) -> None:
        super().init_turn(move=move)
        player, opponent = self.players
        state = self.get_state()
        self.finished = state != "normal"

        if state == "checkmate":
            msg = f"""
Игра окончена: шах и мат!
Победитель: {self.db.get_name(opponent)}
Ходов: {self.board.turn - 1}.
            """
        elif state == "50-move-draw":
            msg = f"""
Игра окончена: за последние 50 ходов не было убито ни одной фигуры и не сдвинуто ни одной пешки.
Ничья!
Ходов: {self.board.turn - 1}
            """
        elif state == "3fold-repetition-draw":
            msg = f"""
Игра окончена: одинаковая позиция доски возникла 3-ий раз подряд.
Ничья!
Ходов: {self.board.turn - 1}
            """
        elif state == "stalemate-draw":
            msg = f"""
Игра окончена: у {"белых" if self.board.is_white_turn else "черных"} не осталось ходов, но их король не под шахом.
Ничья!
Ходов: {self.board.turn - 1}
            """
        else:
            msg = f"Ход {self.board.turn}"
            if move:
                msg += f"\n{move.piece.name}{' -> '+move.new_piece.name if move.is_promotion else ''}"
                msg += f": {encode_pos(move.src)} -> {encode_pos(move.dst)}"
                if move.is_castling:
                    msg += f" ({'Короткая' if move.rook_src.column == 7 else 'Длинная'} рокировка)"
                if move.is_killing:
                    msg += f"\n{move.killed.name} на {encode_pos(move.dst)} убит"
                    msg += f"{'а' if move.killed.name in ['Пешка', 'Ладья'] else ''}!"
                else:
                    msg += "\n"
            else:
                msg += "\n\n"

            if state == "check":
                msg += "\nИгроку поставлен шах!"
            else:
                msg += "\n"

            msg += f"\nХодит { self.db.get_name(player) }; выберите действие:"

        if self.finished:
            video, thumb = self.get_video()
            self.msg.edit_media(
                media=InputMediaVideo(
                    video,
                    caption=msg,
                    filename=self.video_filename,
                    thumb=thumb,
                )
            )
        else:
            keyboard = self._keyboard(
                [
                    {"text": "Ходить", "data": ["TURN"]},
                    {"text": "Сдаться", "data": ["SURRENDER"]},
                ],
                player.id,
            )
            self.init_msg_text = msg
            img = self.visualise_board()
            if self.msg:
                self.msg = self.msg.edit_media(
                    media=InputMediaPhoto(
                        img,
                        caption=msg,
                        filename=self.image_filename,
                    ),
                    reply_markup=keyboard,
                )
            else:
                self.msg = self.bot.send_photo(
                    self.chat_id,
                    img,
                    caption=msg,
                    filename=self.image_filename,
                    reply_markup=keyboard,
                )

    def handle_input(self, args: list[Union[str, int]]) -> None:
        player, opponent = self.players
        allies, _ = self.pieces
        if args[0] == "INIT_MSG":
            self.msg = self.msg.edit_caption(
                self.init_msg_text,
                reply_markup=self._keyboard(
                    [
                        {"text": "Ходить", "data": ["TURN"]},
                        {"text": "Сдаться", "data": ["SURRENDER"]},
                    ],
                    player.id,
                ),
            )

        if args[0] == "TURN":
            piece_buttons = [{"text": "Назад", "data": ["INIT_MSG"]}]
            for piece in allies:
                if next(filter(self.board.is_legal, piece.get_moves()), None):
                    piece_buttons.append(
                        {"text": str(piece), "data": ["CHOOSE_PIECE", piece.pos]}
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = f"Ходит {self.db.get_name(player)}; выберите фигуру:"

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    self.visualise_board(),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(piece_buttons, player.id, head_item=True),
            )

        elif args[0] == "SURRENDER":
            self.finished = True
            video, thumb = self.get_video()
            self.msg = self.msg.edit_media(
                media=InputMediaVideo(
                    video,
                    caption=f"""
Игра окончена: {self.db.get_name(player)} сдался.
Победитель: {self.db.get_name(opponent)}.
Ходов: {self.board.turn - 1}.
                    """,
                    filename=self.video_filename,
                    thumb=thumb,
                )
            )

        elif args[0] == "CHOOSE_PIECE":
            args[1] = decode_pos(args[1])
            dest_buttons = [{"text": "Назад", "data": ["TURN"]}]
            piece = self.board[args[1]]
            moves = list(filter(self.board.is_legal, piece.get_moves()))
            for move in moves:
                if move.is_promotion:
                    dest_buttons.append(
                        {
                            "text": "⏫" + encode_pos(move.dst),
                            "data": ["PROMOTION_MENU", args[1], move.dst],
                        }
                    )
                else:
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + encode_pos(move.dst),
                            "data": ["MOVE", move.pgn],
                        }
                    )
            new_text = self.init_msg_text.split("\n")
            new_text[
                -1
            ] = f"Ходит {self.db.get_name(player)}; выберите новое место фигуры:"

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    self.visualise_board(selected=args[1], possible_moves=moves),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(dest_buttons, player.id, head_item=True),
            )

        elif args[0] == "PROMOTION_MENU":
            move = self.board[args[1]].create_move(args[2], new_piece=Queen)
            pieces = [
                {"text": "Ферзь", "data": ["MOVE", move.pgn]},
                {"text": "Конь", "data": ["MOVE", move.copy(new_piece=Knight).pgn]},
                {"text": "Слон", "data": ["MOVE", move.copy(new_piece=Bishop).pgn]},
                {"text": "Ладья", "data": ["MOVE", move.copy(new_piece=Rook).pgn]},
            ]
            new_text = self.init_msg_text.split("\n")
            new_text[
                -1
            ] = f"Ходит {self.db.get_name(player)}; выберите фигуру, в которую првератится пешка:"

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    self.visualise_board(selected=args[1], possible_moves=[move]),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(pieces, player.id),
            )

        elif args[0] == "MOVE":
            self.init_turn(move=Move.from_pgn(args[1], self.board))


class PMMatch(BaseMatch):
    def __init__(self, player1: User, player2: User, chat1: int, chat2: int, **kwargs):
        self.player1: User = player1
        self.player2: User = player2
        self.chat_id1: int = chat1
        self.chat_id2: int = chat2
        self.msg1: Message = None
        self.msg2: Message = None
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, obj: JSON, match_id: int, bot=Bot) -> "PMMatch":
        logging.debug(f"Constructing {cls.__name__} object:", obj)
        player1 = User.de_json(obj["player1"] | {"is_bot": False}, bot)
        player2 = User.de_json(obj["player2"] | {"is_bot": False}, bot)
        new = cls(
            player1,
            player2,
            obj["chat_id1"],
            obj["chat_id2"],
            bot=bot,
            id=match_id,
        )
        new.states = [BoardInfo.from_fen(fen) for fen in obj["states"]]
        new.init_msg_text = obj["msg_text"]
        new.msg1 = Message(
            obj["msg_id1"],
            time.monotonic,
            Chat(obj["chat_id1"], "private", bot=bot),
            bot=bot,
            caption=obj["msg_text"],
        )
        new.msg2 = Message(
            obj["msg_id2"],
            time.monotonic,
            Chat(obj["chat_id2"], "private", bot=bot),
            bot=bot,
        )
        new.turn += 1
        new.empty_halfturns += 1
        new.is_white_turn = not new.is_white_turn

        return new

    @property
    def player_msg(self) -> Message:
        return self.msg1 if self.board.is_white_turn else self.msg2

    @player_msg.setter
    def player_msg(self, msg: Message) -> None:
        if self.board.is_white_turn:
            self.msg1 = msg
        else:
            self.msg2 = msg

    @property
    def opponent_msg(self) -> Message:
        return self.msg2 if self.board.is_white_turn else self.msg1

    @opponent_msg.setter
    def opponent_msg(self, msg: Message) -> None:
        if self.board.is_white_turn:
            self.msg2 = msg
        else:
            self.msg1 = msg

    @property
    def players(self) -> tuple[User, User]:
        return (
            (self.player1, self.player2)
            if self.board.is_white_turn
            else (self.player2, self.player1)
        )

    @property
    def chat_ids(self) -> tuple[int, int]:
        return (
            (self.chat_id1, self.chat_id2)
            if self.board.is_white_turn
            else (self.chat_id2, self.chat_id1)
        )

    def to_dict(self) -> JSON:
        res = super().to_dict()
        res.update(
            {
                "type": "PM",
                "chat_id1": self.chat_id1,
                "chat_id2": self.chat_id2,
                "msg_id1": self.msg1.message_id,
                "msg_id2": getattr(self.msg2, "message_id", None),
                "msg_text": self.init_msg_text,
                "player1": {
                    k: v
                    for k, v in self.player1.to_dict().items()
                    if k in ["username", "id", "first_name", "last_name"]
                },
                "player2": {
                    k: v
                    for k, v in self.player2.to_dict().items()
                    if k in ["username", "id", "first_name", "last_name"]
                },
            }
        )
        return res

    def init_turn(self, move: Move = None, call_parent_method: bool = True):
        if call_parent_method:
            super().init_turn(move=move)
        player, opponent = self.players
        player_chatid, opponent_chatid = self.chat_ids
        state = self.get_state()
        self.finished = state != "normal"

        if state == "checkmate":
            player_text = opponent_text = f"""
Игра окончена: шах и мат!
Победитель: {self.db.get_name(opponent)}
Ходов: {self.board.turn - 1}.
            """
        elif state == "50-move-draw":
            player_text = opponent_text = f"""
Игра окончена: за последние 50 ходов не было убито ни одной фигуры и не сдвинуто ни одной пешки.
Ничья!
Ходов: {self.board.turn - 1}
            """
        elif state == "3fold-repetition-draw":
            player_text = opponent_text = f"""
Игра окончена: одинаковая позиция доски возникла 3-ий раз подряд.
Ничья!
Ходов: {self.board.turn - 1}
            """
        elif state == "stalemate-draw":
            player_text = opponent_text = f"""
Игра окончена: у {"белых" if self.board.is_white_turn else "черных"} не осталось ходов, но их король не под шахом.
Ничья!
Ходов: {self.board.turn - 1}
            """
        else:
            player_text = f"Ход {self.board.turn}"
            if move:
                player_text += f"\n{move.piece.name}{' -> '+move.new_piece.name if move.is_promotion else ''}"
                player_text += f": {encode_pos(move.src)} -> {encode_pos(move.dst)}"
                if move.is_castling:
                    player_text += f' ({"Короткая" if  move.rook_src.column == 7 else "Длинная"} рокировка)'
                if move.is_killing:
                    player_text += f"\n{move.killed.name} на {encode_pos(move.dst)} игрока {self.db.get_name(player)} убит"
                    player_text += (
                        f"{'а' if move.killed.name in ['Пешка', 'Ладья'] else ''}!"
                    )
                else:
                    player_text += "\n"
            else:
                player_text += "\n\n"

            if state == "check":
                player_text += f"\nИгроку {self.db.get_name(player)} поставлен шах!"
            else:
                player_text += "\n"

            opponent_text = player_text

            player_text += "\nВыберите действие:"
            opponent_text += f"\nХодит {self.db.get_name(player)}"

        if self.finished:
            video, thumb = self.get_video()
            new_msg = InputMediaVideo(
                video, caption=player_text, filename=self.video_filename, thumb=thumb
            )
            self.player_msg = self.player_msg.edit_media(media=new_msg)
            if self.opponent_msg:
                self.opponent_msg = self.opponent_msg.edit_media(media=new_msg)
        else:
            self.init_msg_text = player_text
            keyboard = self._keyboard(
                [
                    {"text": "Ходить", "data": ["TURN"]},
                    {"text": "Сдаться", "data": ["SURRENDER"]},
                ],
                player.id,
            )
            if self.player_msg:
                self.player_msg = self.player_msg.edit_media(
                    media=InputMediaPhoto(
                        self.visualise_board(),
                        caption=player_text,
                        filename=self.image_filename,
                    ),
                    reply_markup=keyboard,
                )
            else:
                self.player_msg = self.bot.send_photo(
                    player_chatid,
                    self.visualise_board(),
                    caption=player_text,
                    filename=self.image_filename,
                    reply_markup=keyboard,
                )

            if opponent_chatid:
                if self.player_msg:
                    self.opponent_msg = self.opponent_msg.edit_media(
                        media=InputMediaPhoto(
                            self.visualise_board(),
                            caption=opponent_text,
                            filename=self.image_filename,
                        )
                    )
                else:
                    self.opponent_msg = self.bot.send_photo(
                        opponent_chatid,
                        self.visualise_board(),
                        caption=opponent_text,
                        filename=self.image_filename,
                    )

    def handle_input(self, args):
        player, opponent = self.players
        allies, _ = self.pieces

        if args[0] == "INIT_MSG":
            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    self.visualise_board(),
                    caption=self.init_msg_text,
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(
                    [
                        {"text": "Ходить", "data": ["TURN"]},
                        {"text": "Сдаться", "data": ["SURRENDER"]},
                    ],
                    player.id,
                ),
            )

        if args[0] == "TURN":
            piece_buttons = [{"text": "Назад", "data": ["INIT_MSG"]}]
            for piece in allies:
                if next(
                    filter(self.board.is_legal, piece.get_moves()),
                    None,
                ):
                    piece_buttons.append(
                        {"text": str(piece), "data": ["CHOOSE_PIECE", piece.pos]}
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = f"Выберите фигуру:"

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    self.visualise_board(),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(piece_buttons, player.id, head_item=True),
            )

        elif args[0] == "SURRENDER":
            self.finished = True
            video, thumb = self.get_video()
            for msg in [self.msg1, self.msg2]:
                if msg:
                    msg.edit_media(
                        media=InputMediaVideo(
                            video,
                            caption=f"""
Игра окончена: {self.db.get_name(player)} сдался.
Победитель: {self.db.get_name(opponent)}.
Ходов: {self.board.turn - 1}.
                            """,
                            filename=self.video_filename,
                            thumb=thumb,
                        )
                    )

        elif args[0] == "CHOOSE_PIECE":
            args[1] = decode_pos(args[1])
            dest_buttons = [{"text": "Назад", "data": ["TURN"]}]
            piece = self.board[args[1]]
            moves = list(filter(self.board.is_legal, piece.get_moves()))
            for move in moves:
                if move.is_promotion:
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + encode_pos(move.dst),
                            "data": ["PROMOTION_MENU", args[1], move.dst],
                        }
                    )
                else:
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + encode_pos(move.dst),
                            "data": ["MOVE", move.pgn],
                        }
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = f"Выберите новое место фигуры:"

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    self.visualise_board(selected=args[1], possible_moves=moves),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(dest_buttons, player.id, head_item=True),
            )

        elif args[0] == "PROMOTION_MENU":
            args[1] = decode_pos(args[1])
            args[2] = decode_pos(args[2])
            move = self.board[args[1]].create_move(args[2], new_piece=Queen)
            pieces = [
                {"text": "Ферзь", "data": ["PROMOTION", move.pgn]},
                {
                    "text": "Конь",
                    "data": ["PROMOTION", move.copy(new_piece=Knight).pgn],
                },
                {
                    "text": "Слон",
                    "data": ["PROMOTION", move.copy(new_piece=Bishop).pgn],
                },
                {"text": "Ладья", "data": ["PROMOTION", move.copy(new_piece=Rook).pgn]},
            ]

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = f"Выберите фигуру, в которую првератится пешка:"

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    self.visualise_board(selected=args[1], possible_moves=[move]),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(pieces, player.id),
            )

        elif args[0] == "MOVE":
            return self.init_turn(move=Move.from_pgn(args[1], self.board))


class AIMatch(PMMatch):
    def __init__(self, player: User, chat_id: int, player2: User = None, **kwargs):
        ai_player = player2 if player2 else kwargs["bot"].get_me()
        self.ai_rating: int = None
        super().__init__(player, ai_player, chat_id, 0, **kwargs)
        self.engine_api = subprocess.Popen(
            os.environ["ENGINE_FILENAME"],
            bufsize=1,
            universal_newlines=True,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        self.engine_api.stdout.readline()
        self.engine_api.stdin.write("setoption name UCI_LimitStrength value true\n")

    @classmethod
    def from_dict(cls, obj: JSON, match_id: int, bot=Bot) -> "AIMatch":
        logging.debug(f"Constructing {cls.__name__} object: {obj}")
        player = User.de_json(obj["player1"] | {"is_bot": False}, bot)
        new = cls(player, obj["chat_id1"], bot=bot, id=match_id)
        new.states = [BoardInfo.from_fen(fen) for fen in obj["states"]]
        new.init_msg_text = obj["msg_text"]
        new.set_elo(obj["ai_rating"])
        new.msg1 = Message(
            obj["msg_id1"],
            time.monotonic,
            Chat(obj["chat_id1"], "private", bot=bot),
            bot=bot,
            caption=obj["msg_text"],
        )

        return new

    def to_dict(self) -> JSON:
        res = super().to_dict()
        del res["player2"]
        del res["msg_id2"]
        del res["chat_id2"]
        res.update({"ai_rating": self.ai_rating, "type": "AI"})
        return res

    def set_elo(self, value: int) -> None:
        self.ai_rating = value
        self.engine_api.stdin.write(f"setoption name UCI_Elo value {value}\n")

    def init_turn(self, setup: bool = False, **kwargs) -> None:
        if setup:
            self.msg1 = self.bot.send_photo(
                self.chat_id1,
                self.visualise_board(),
                caption="Выберите уровень сложности:",
                filename=self.image_filename,
                reply_markup=self._keyboard(
                    [
                        {"text": "Низкий", "data": ["SKILL_LEVEL", "1350"]},
                        {"text": "Средний", "data": ["SKILL_LEVEL", "1850"]},
                        {"text": "Высокий", "data": ["SKILL_LEVEL", "2350"]},
                        {"text": "Легендарный", "data": ["SKILL_LEVEL", "2850"]},
                    ],
                    self.player1.id,
                ),
            )

        else:
            turn_info = BaseMatch.init_turn(self, **kwargs)
            if self.finished:
                return super().init_turn(self, call_parent_method=False, **kwargs)

            self.engine_api.stdin.write(f"position fen {self.board.fen}\n")
            self.engine_api.stdin.write(f"go depth 2\n")
            for line in self.engine_api.stdout:
                if "bestmove" in line:
                    turn = line.split(" ")[1].strip("\n")
                    break
            return super().init_turn(
                move=self.board[decode_pos(turn[:2])].create_move(
                    decode_pos(turn[2:4]),
                    new_piece=eval(FENSYMBOLS[turn[-1]]) if len(turn) == 5 else None
                ),
            )

    def handle_input(self, args):
        if args[0] == "SKILL_LEVEL":
            self.set_elo(args[1])
            return super().init_turn()
        else:
            return super().handle_input(args)
