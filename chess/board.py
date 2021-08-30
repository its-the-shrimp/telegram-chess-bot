from typing import Type, Optional
from . import pieces
from .base import *

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

def _decode_fen(fen: str) -> dict[tuple, str]:
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
                pieces.King,
                rook_src=BoardPoint(7, row),
                rook_dst=BoardPoint(5, row),
            )
        elif move == "O-O-O":
            return cls(
                board_obj,
                BoardPoint(4, row),
                BoardPoint(2, row),
                pieces.King,
                rook_src=BoardPoint(0, row),
                rook_dst=BoardPoint(3, row),
            )
        else:
            if move[0] in "abcdefghx":
                piece_cls = pieces.Pawn
            else:
                piece_cls = getattr(pieces, symbols[move[0]])
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
                promoted_to = getattr(pieces, symbols[promoted_to])
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

            dst = decode_pos(move)
            src = piece_cls.get_prev_pos(
                dst, 
                board_obj.is_white_turn, 
                board_obj, 
                column_hint=column_hint,
                row_hint=row_hint
                )
        
        return cls(
            board_obj,
            src,
            dst,
            piece_cls,
            killed=type(board_obj[dst]),
            new_piece=promoted_to,
        )

    def __init__(
        self,
        board_obj: "BoardInfo",
        src: BoardPoint,
        dst: BoardPoint,
        piece: Type["pieces.BasePiece"],
        killed: Optional[Type["pieces.BasePiece"]] = type(None),
        rook_src: BoardPoint = None,
        rook_dst: BoardPoint = None,
        new_piece: type = None,
    ):
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
        return self.piece == pieces.King and self.rook_src

    @property
    def is_promotion(self) -> bool:
        return self.piece == pieces.Pawn and bool(self.new_piece)

    @property
    def enpassant_pos(self) -> list[Optional[BoardPoint]]:
        if self.piece == pieces.Pawn and abs(self.src[1] - self.dst[1]) == 2:
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
        king = self.board[pieces.King, not self.board.is_white_turn][0]
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
        move = self.piece.get_pos_hints(self) + move

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

        for (column, line), char in _decode_fen(board).items():
            new = getattr(pieces, FENSYMBOLS[char.lower()])(
                BoardPoint(column, line), None, char.isupper()
            )
            K = "K" if new.is_white else "k"
            Q = "Q" if new.is_white else "q"
            if type(new) == pieces.King and K not in castlings and Q not in castlings:
                new.moved = True
            elif (
                type(new) == pieces.Rook
                and K not in castlings
                and Q in castlings
                and new.pos != [0, 0 if new.is_white else 7]
            ):
                new.moved = True
            elif (
                type(new) == pieces.Rook
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
        board: list["pieces.BasePiece"],
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

        src_piece = sorted(
            src_pieces, key=lambda x: int(x.is_white == other.is_white_turn)
        )[-1]
        return src_piece.create_move(
            dst_pieces[0].pos,
            new_piece=type(dst_pieces[0])
            if type(dst_pieces[0]) != type(src_piece)
            else None,
        )

    def __add__(self, move: Move) -> "BoardInfo":
        new = self.copy(
            turn=self.turn + (1 if not self.is_white_turn else 0),
            is_white_turn=not self.is_white_turn,
            empty_halfturns=0
            if move.is_killing or move.piece == pieces.Pawn
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
        new.enpassant_pos = move.enpassant_pos

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
    def whites(self) -> list["pieces.BasePiece"]:
        return list(filter(lambda x: x.is_white, self.board))

    @property
    def blacks(self) -> list["pieces.BasePiece"]:
        return list(filter(lambda x: not x.is_white, self.board))

    @property
    def castlings(self) -> str:
        res = ""
        white_king = self[pieces.King, True][0]
        black_king = self[pieces.King, False][0]
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
        return type(self)(
            [piece.copy() for piece in self.board], **(params | new_params)
        )

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
        return not test_obj[pieces.King, self.is_white_turn][0].in_check()