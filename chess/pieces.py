from __future__ import annotations

from typing import Callable, Union, Type, List, TYPE_CHECKING
from .base import *
from .board import Move
import itertools

if TYPE_CHECKING:
    from board import BoardInfo

class BasePiece:
    name: str
    fen_symbol: list[str]
    move_diffs: Union[tuple[list[int]], tuple[tuple[list[int]]]]
    value: int

    def __init__(self, pos: BoardPoint, board: "BoardInfo", is_white: bool):
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

    @classmethod
    def _get_moves(cls, pos: BoardPoint, is_white: bool, board: BoardInfo, all: bool=False, **kwargs) -> list[BoardPoint]:
        moves = []
        allies = [i.pos for i in board.board if i.is_white == is_white]
        for diff in cls.move_diffs:
            move_dst = BoardPoint(*[i + d for i, d in zip(pos, diff)])
            if in_bounds(move_dst) and (move_dst not in allies or all):
                moves.append(move_dst)

        return moves

    @classmethod
    def get_prev_pos(
        cls, 
        pos: BoardPoint,
        is_white: bool, 
        board: BoardInfo,
        column_hint: int = None,
        row_hint: int = None,
        **kwargs
    ) -> BoardPoint:
        possible_pos = []
        for move in cls._get_moves(pos, is_white, board, reverse=cls==Pawn, all=True):
            piece = board[move]
            if all([
                type(piece) == cls,
                getattr(piece, "is_white", None) == is_white,
                move.row == row_hint or row_hint is None,
                move.column == column_hint or column_hint is None
            ]):
                possible_pos.append(move)

        assert len(possible_pos) == 1
        return possible_pos[0]

    @classmethod
    def get_pos_hints(cls, move: Move):
        possible_positions = [i for i in cls._get_moves(
            move.dst, 
            move.board.is_white_turn, 
            move.board, 
            reverse=cls==Pawn,
            all=True
        ) if type(move.board[i]) == cls]
        if len(possible_positions) < 2:
            return ""
        elif len([i.column for i in possible_positions if i.column == move.src.column]) > 1:
            return str(move.src.row)
        else:
            return chr(move.src.column + 97)

    def get_moves(self, **kwargs) -> list[Move]:
        moves = self._get_moves(self.pos, self.is_white, self.board)
        return [self.create_move(dst) for dst in moves]

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
    move_diffs = ([0, 1], [0, 2], [1, 1], [-1, 1])
    value = 1

    @classmethod
    def _get_moves(
            cls,
            pos: BoardPoint,
            is_white: bool,
            board: BoardInfo,
            reverse: bool = False,
            all: bool = False
        ) -> list[BoardPoint]:
        positions = []
        diffs = [[i[0], i[1] * (1 if is_white else -1)] for i in cls.move_diffs]
        if reverse:
            is_white = not is_white
            if board[pos]:
                diffs = [[i[0], i[1] * -1] for i in diffs[2:]]
            else:
                diffs = [[i[0], i[1] * -1] for i in diffs[:2]]

        for diff in diffs:
            move_dst = BoardPoint(*[i + d for i, d in zip(pos, diff)])
            if in_bounds(move_dst):
                if reverse:
                    if any([
                        abs(diff[0]) == 1 and getattr(board[move_dst], "is_white", None) == is_white,
                        diff[1] == -2 and not board[move_dst._replace(column=move_dst.column - 1)] and board[move_dst],
                        board[move_dst]
                    ]):
                        positions.append(move_dst)
                else:
                    if any([
                        abs(diff[0]) == 1 and getattr(board[move_dst], "is_white", is_white) != is_white,
                        diff[1] == 2 and not board[move_dst._replace(column=move_dst.column - 1)] and not board[move_dst],
                        abs(diff[0]) == 0 and not board[move_dst]
                    ]):
                        positions.append(move_dst)
        return positions

        


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
    def _get_moves(cls, pos: BoardPoint, is_white: bool, board: BoardInfo, all: bool = False, **kwargs) -> list[BoardPoint]:
        moves = []
        allies_pos = [i.pos for i in board[None, is_white]]
        enemies_pos = [i.pos for i in board[None, not is_white]]  
        for diff_seq in cls.move_diffs:
            for diff in diff_seq:
                move = BoardPoint(*[i + d for i, d in zip(pos, diff)])
                if (move in allies_pos and not all) or not in_bounds(move):
                    break
                else:
                    moves.append(move)
                    if move in enemies_pos or (all and move in allies_pos):
                        break

        return moves


class Bishop(Rook):
    name = "Слон"
    fen_symbol = ["b", "B"]
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
    move_diffs = Rook.move_diffs + Bishop.move_diffs
    value = 8


class King(BasePiece):
    name = "Король"
    fen_symbol = ["k", "K"]
    move_diffs = tuple(i for i in itertools.product([1, 0, -1], [1, 0, -1]) if i != (0, 0))
    value = 99

    def get_moves(self, for_fen: bool = False, castling: bool = True) -> list[Move]:
        moves = self._get_moves(self.pos, self.is_white, self.board)

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