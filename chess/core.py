import itertools
import random
from typing import Iterator, Type, Optional, Union
from .utils import BoardPoint, _reversed
import enum

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
    "ru": {
        "Pawn": "",
        "Rook": "Л",
        "Bishop": "С",
        "Knight": "К",
        "Queen": "Ф",
        "King": "Кр",
    },
}


class MoveEval(enum.Enum):
    BLUNDER = "??"
    MISTAKE = "?"
    WEAK = "?!"
    FORCED = "□"
    GOOD = "!"
    GREAT = "!!"
    BEST = "!!!"
    PRECISE = "!?"


class GameState(enum.Enum):
    NORMAL = enum.auto()
    CHECK = enum.auto()
    ABORTED = enum.auto()
    WHITE_RESIGNED = enum.auto()
    BLACK_RESIGNED = enum.auto()
    WHITE_CHECKMATED = enum.auto()
    BLACK_CHECKMATED = enum.auto()
    DRAW = enum.auto()
    THREEFOLD_REPETITION_DRAW = enum.auto()
    STALEMATE_DRAW = enum.auto()
    INSUFFIECENT_MATERIAL_DRAW = enum.auto()
    FIFTY_MOVE_DRAW = enum.auto()


class BasePiece:
    fen_symbol: list[str]
    move_diffs: Union[tuple[list[int]], tuple[tuple[list[int]]]]
    value: int

    def __init__(self, pos: BoardPoint, board: "BoardInfo", is_white: bool):
        self.pos = pos
        self.is_white = is_white
        self.board = board
        self.fen_symbol = self.fen_symbol[int(is_white)]
        self.promoted = False

    def __str__(self):
        return f"{self.__class__.__name__} on {str(self.pos)}"

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

    @classmethod
    def _get_moves(
        cls,
        pos: BoardPoint,
        is_white: bool,
        board: "BoardInfo",
        all: bool = False,
        **kwargs,
    ) -> list[BoardPoint]:
        moves = []
        allies = [i.pos for i in board.board if i.is_white == is_white]
        for diff in cls.move_diffs:
            move_dst = BoardPoint(*[i + d for i, d in zip(pos, diff)])
            if move_dst and (move_dst not in allies or all):
                moves.append(move_dst)

        return moves

    @classmethod
    def get_prev_pos(
        cls,
        pos: BoardPoint,
        is_white: bool,
        board: "BoardInfo",
        column_hint: int = None,
        row_hint: int = None,
        **kwargs,
    ) -> BoardPoint:
        possible_pos = []
        for move in cls._get_moves(pos, is_white, board, reverse=cls == Pawn, all=True):
            piece = board[move]
            if all(
                [
                    type(piece) == cls,
                    getattr(piece, "is_white", None) == is_white,
                    move.rank == row_hint or row_hint is None,
                    move.file == column_hint or column_hint is None,
                ]
            ):
                possible_pos.append(move)

        try:
            assert len(possible_pos) == 1
        except AssertionError as exc:
            print(possible_pos)
            raise exc
        return possible_pos[0]

    @classmethod
    def get_pos_hints(cls, move: "Move") -> str:
        possible_positions = []
        for pos in cls._get_moves(
            move.dst,
            move.board.is_white_turn,
            move.board,
            reverse=cls == Pawn,
            all=True,
        ):
            piece = move.board[pos]
            if type(piece) == cls and piece.is_white == move.board.is_white_turn:
                possible_positions.append(pos)

        if len(possible_positions) < 2 and not (cls == Pawn and move.is_capturing()):
            return ""
        elif len([i.file for i in possible_positions if i.file == move.src.file]) > 1:
            return str(move.src.rank)
        else:
            return chr(move.src.file + 97)

    def _move(self, pos: BoardPoint) -> None:
        if isinstance(self, King):
            self.kingside_rook_pos = None
            self.queenside_rook_pos = None
        elif isinstance(self, Rook):
            allied_king = self.board[King, self.is_white][0]

            if self.pos == allied_king.kingside_rook_pos:
                allied_king.kingside_rook_pos = None
            elif self.pos == allied_king.queenside_rook_pos:
                allied_king.queenside_rook_pos = None

        self.pos = pos

    def get_moves(self, **kwargs) -> list["Move"]:
        moves = [
            Move.from_piece(self, dst)
            for dst in self._get_moves(self.pos, self.is_white, self.board, **kwargs)
        ]
        return list(filter(Move.is_legal, moves))

    def copy(self) -> "BasePiece":
        new = type(self)(self.pos, self.board, self.is_white)
        if isinstance(self, King):
            new.queenside_rook_pos = self.queenside_rook_pos
            new.kingside_rook_pos = self.kingside_rook_pos
        return new


class Pawn(BasePiece):
    fen_symbol = ["p", "P"]
    move_diffs = ([0, 1], [0, 2], [1, 1], [-1, 1])
    value = 1

    @classmethod
    def _get_moves(
        cls,
        pos: BoardPoint,
        is_white: bool,
        board: "BoardInfo",
        reverse: bool = False,
        all: bool = False,
        only_capturing: bool = False,
        **kwargs,
    ) -> list[BoardPoint]:
        positions = []
        diffs = [[i[0], i[1] * (1 if is_white else -1)] for i in cls.move_diffs]
        if reverse:
            is_white = not is_white
            if board[pos] or board.enpassant_pos[1] == pos or only_capturing:
                diffs = [[i[0], i[1] * -1] for i in diffs[2:]]
            else:
                diffs = [[i[0], i[1] * -1] for i in diffs[:2]]
                if pos.rank != (4 if is_white else 3):
                    del diffs[1]
        elif only_capturing:
            diffs = diffs[2:]
        elif pos.rank != (1 if is_white else 6):
            del diffs[1]
        direction = 1 if is_white else -1

        for diff in diffs:
            move_dst = BoardPoint(*[i + d for i, d in zip(pos, diff)])
            if move_dst:
                existing_piece = board[move_dst]
                if any(
                    [
                        abs(diff[0]) == 1
                        and (
                            getattr(existing_piece, "is_white", is_white) != is_white
                            or all
                        ),
                        abs(diff[0]) == 1
                        and board.enpassant_pos[1] == (pos if reverse else move_dst),
                        diff[1] == direction * 2
                        and not board[move_dst.copy(rank=move_dst.rank - direction)]
                        and bool(existing_piece) == reverse,
                        abs(diff[0]) == 0
                        and diff[1] == direction
                        and bool(existing_piece) == reverse,
                    ]
                ):
                    positions.append(move_dst)

        return positions


class Knight(BasePiece):
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


class Bishop(BasePiece):
    fen_symbol = ["b", "B"]
    move_diffs = (
        tuple(zip(range(1, 8), range(1, 8))),
        tuple(zip(range(-1, -8, -1), range(-1, -8, -1))),
        tuple(zip(range(-1, -8, -1), range(1, 8))),
        tuple(zip(range(1, 8), range(-1, -8, -1))),
    )
    value = 3

    @classmethod
    def _get_moves(
        cls,
        pos: BoardPoint,
        is_white: bool,
        board: "BoardInfo",
        all: bool = False,
        **kwargs,
    ) -> list[BoardPoint]:
        moves = []
        allies_pos = [i.pos for i in board[None, is_white]]
        enemies_pos = [i.pos for i in board[None, not is_white]]
        for diff_seq in cls.move_diffs:
            for diff in diff_seq:
                move = BoardPoint(*[i + d for i, d in zip(pos, diff)])
                if (move in allies_pos and not all) or not move:
                    break
                else:
                    moves.append(move)
                    if move in enemies_pos or move in allies_pos:
                        break

        return moves


class Rook(Bishop):
    fen_symbol = ["r", "R"]
    move_diffs = (
        tuple(zip(range(1, 8), [0] * 7)),
        tuple(zip(range(-1, -8, -1), [0] * 7)),
        tuple(zip([0] * 7, range(1, 8))),
        tuple(zip([0] * 7, range(-1, -8, -1))),
    )
    value = 5


class Queen(Bishop):
    name = "Ферзь"
    fen_symbol = ["q", "Q"]
    move_diffs = Rook.move_diffs + Bishop.move_diffs
    value = 8


class King(BasePiece):
    name = "Король"
    fen_symbol = ["k", "K"]
    move_diffs = tuple(
        i for i in itertools.product([1, 0, -1], [1, 0, -1]) if i != (0, 0)
    )
    value = 99

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queenside_rook_pos: BoardPoint = None
        self.kingside_rook_pos: BoardPoint = None

    def get_moves(self, castling: bool = True, **kwargs) -> list["Move"]:
        moves = super().get_moves()

        if castling and not self.in_check():
            Y = 0 if self.is_white else 7

            if self.queenside_rook_pos is not None:
                moves.append(Move.from_piece(self, BoardPoint(2, Y)))
                for mid_column in range(
                    min(2, self.queenside_rook_pos.file + 1), self.pos.file
                ):
                    mid_pos = BoardPoint(mid_column, Y)
                    if (
                        self._in_check(mid_pos, self.is_white, self.board)
                        or self.board[mid_pos]
                    ):
                        del moves[-1]
                        break

            if self.kingside_rook_pos is not None:
                moves.append(Move.from_piece(self, BoardPoint(6, Y)))
                for mid_column in range(
                    self.pos.file + 1, max(7, self.kingside_rook_pos.file)
                ):
                    mid_pos = BoardPoint(mid_column, Y)
                    if (
                        self._in_check(mid_pos, self.is_white, self.board)
                        or self.board[mid_pos]
                    ):
                        del moves[-1]
                        break

        return moves

    def in_checkmate(self):
        allied_moves = itertools.chain(
            *[piece.get_moves() for piece in self.allied_pieces]
        )

        return next(allied_moves, None) is None and self.in_check()

    @classmethod
    def _in_check(_cls, pos: BoardPoint, is_white: bool, board: "BoardInfo") -> bool:
        for cls in (Rook, Bishop):
            for move_seq in cls.move_diffs:
                for diff in move_seq:
                    piece = board[BoardPoint(*[i + d for i, d in zip(diff, pos)])]
                    if type(piece) in (cls, Queen) and piece.is_white != is_white:
                        return True
                    elif piece is not None:
                        break

        for cls in (King, Knight):
            for diff in cls.move_diffs:
                piece = board[BoardPoint(*[i + d for i, d in zip(diff, pos)])]
                if type(piece) == cls and piece.is_white != is_white:
                    return True

        for diff in ((1, 1 if is_white else -1), (-1, 1 if is_white else -1)):
            piece = board[BoardPoint(*[i + d for i, d in zip(diff, pos)])]
            if type(piece) == Pawn and piece.is_white != is_white:
                return True

        return False

    def in_check(self) -> bool:
        return self._in_check(self.pos, self.is_white, self.board)


class Move:
    PROMOTION_FLAGS = {
        "Knight": "00",
        "Bishop": "01",
        "Rook": "10",
        "Queen": "11",
    }

    @classmethod
    def from_hash(cls, src: int, board: "BoardInfo") -> "Move":
        src = f"{src:0>16b}"[:-1]
        src_column, src_row, dst_column, dst_row = (
            int(src[i * 3 : i * 3 + 3], 2) for i in range(4)
        )
        promotion = src[-3:]
        if promotion[0] == "1":
            new_piece = eval(_reversed(cls.PROMOTION_FLAGS)[promotion[1:]]).fen_symbol[
                0
            ]
        else:
            new_piece = ""
        return Move.from_piece(
            board[src_column, src_row],
            BoardPoint(dst_column, dst_row),
            new_piece=new_piece,
        )

    @classmethod
    def from_piece(
        cls, piece: BasePiece, dst: BoardPoint, new_piece: Type[BasePiece] = None
    ) -> "Move":
        if isinstance(piece, King) and dst.file in (2, 6):
            return cls(
                piece.board,
                piece.is_white,
                piece.pos,
                dst,
                type(piece),
                rook_src=piece.kingside_rook_pos
                if dst.file == 6
                else piece.queenside_rook_pos,
            )
        if piece.board.enpassant_pos[1] == dst:
            killed_piece = piece.board[piece.board.enpassant_pos[0]]
        else:
            killed_piece = piece.board[dst]
        killed = (
            type(killed_piece)
            if killed_piece and killed_piece.is_white != piece.is_white
            else None
        )
        return cls(
            piece.board,
            piece.is_white,
            piece.pos,
            dst,
            type(piece),
            new_piece=new_piece,
            killed=killed,
        )

    @classmethod
    def from_pgn(cls, move: str, board_obj: "BoardInfo", language_code: str = "en"):
        symbols = _reversed(PGNSYMBOLS[language_code])
        row = 0 if board_obj.is_white_turn else 7
        move = move.rstrip("#+")

        if move == "O-O":
            allied_king = board_obj[King, board_obj.is_white_turn][0]
            return cls(
                board_obj,
                board_obj.is_white_turn,
                allied_king.pos,
                BoardPoint(6, row),
                King,
                rook_src=allied_king.kingside_rook_pos,
            )
        elif move == "O-O-O":
            allied_king = board_obj[King, board_obj.is_white_turn][0]
            return cls(
                board_obj,
                board_obj.is_white_turn,
                allied_king.pos,
                BoardPoint(2, row),
                King,
                rook_src=allied_king.queenside_rook_pos,
            )
        else:
            if move[0] in "abcdefghx":
                piece_cls = Pawn
            else:
                piece_cls = eval(symbols[move[0]])
                move = move[1:]

            if "x" in move:
                hint, move = move.split("x")
            else:
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

            dst = BoardPoint(move)
            src = piece_cls.get_prev_pos(
                dst,
                board_obj.is_white_turn,
                board_obj,
                column_hint=column_hint,
                row_hint=row_hint,
            )
        if board_obj.enpassant_pos[1] == dst:
            killed_piece = board_obj[board_obj.enpassant_pos[0]]
        else:
            killed_piece = board_obj[dst]

        return cls(
            board_obj,
            board_obj.is_white_turn,
            src,
            dst,
            piece_cls,
            killed=type(killed_piece) if killed_piece else None,
            new_piece=promoted_to,
        )

    def __init__(
        self,
        board_obj: "BoardInfo",
        is_white: bool,
        src: BoardPoint,
        dst: BoardPoint,
        piece: Type[BasePiece],
        killed: Type[BasePiece] = None,
        new_piece: Type[BasePiece] = None,
        rook_src: BoardPoint = None,
        evaluation: MoveEval = "",
    ):
        self.src: BoardPoint = src
        self.dst: BoardPoint = dst
        self.is_white: bool = is_white
        self.piece: Type[BasePiece] = piece
        self.killed: Optional[Type[BasePiece]] = killed
        self.new_piece: Type[BasePiece] = new_piece
        self.board: BoardInfo = board_obj
        self.evaluation: MoveEval = evaluation
        self.rook_src: Optional[BoardPoint] = rook_src

    def __hash__(self):
        res = f"{self.src.file:0>3b}{self.src.rank:0>3b}{self.dst.file:0>3b}{self.dst.rank:0>3b}"
        res += str(int(self.is_promotion()))
        res += self.PROMOTION_FLAGS[
            self.new_piece.__name__ if res[-1] == "1" else "Knight"
        ]
        res += "0"
        return int(res, 2)

    def __repr__(self):
        return f"Move({self.pgn_encode()})"

    def __eq__(self, other: "Move") -> bool:
        return (
            type(self) == type(other)
            and self.board == other.board
            and self.pgn_encode() == other.pgn_encode()
        )

    def is_capturing(self) -> bool:
        return self.killed is not None

    def is_castling(self) -> bool:
        return self.rook_src is not None

    def is_promotion(self) -> bool:
        return self.piece == Pawn and bool(self.new_piece)

    def apply(self) -> "BoardInfo":
        return self.board + self

    def enpassant_pos(self) -> list[Optional[BoardPoint]]:
        if self.piece == Pawn and abs(self.src.rank - self.dst.rank) == 2:
            return [
                self.dst,
                BoardPoint(self.dst.file, (self.src.rank + self.dst.rank) // 2),
            ]
        return [None, None]

    @property
    def capture_dst(self):
        if self.is_capturing():
            return (
                self.board.enpassant_pos[0]
                if self.dst == self.board.enpassant_pos[1]
                else self.dst
            )

    @property
    def type(self) -> str:
        if self.is_capturing() and self.is_promotion():
            return "killing-promotion"
        elif self.is_promotion():
            return "promotion"
        elif self.is_capturing():
            return "killing"
        elif self.is_castling():
            return "castling"
        return "normal"

    @property
    def rook_dst(self) -> Optional[BoardPoint]:
        if not self.is_castling():
            return

        if self.src.file > self.dst.file:
            return BoardPoint(3, self.dst.rank)
        else:
            return BoardPoint(5, self.dst.rank)

    @property
    def pgn_opponent_state(self) -> str:
        test_board = self.board + self
        king = test_board[King, not self.board.is_white_turn][0]
        if king.in_checkmate():
            return "#"
        elif king.in_check():
            return "+"
        return ""

    def pgn_encode(self, language_code: str = "en") -> str:
        if self.is_castling():
            return "-".join(["O"] * abs(self.rook_src.file - self.rook_dst.file))
        move = ("x" if self.is_capturing() else "") + str(self.dst)
        move = self.piece.get_pos_hints(self) + move

        move = PGNSYMBOLS[language_code][self.piece.__name__] + move
        if self.is_promotion():
            move += "=" + PGNSYMBOLS[language_code][self.new_piece.__name__]

        return move + self.pgn_opponent_state + self.evaluation

    def copy(self, **params) -> "Move":
        defaults = {
            "is_white": self.is_white,
            "board_obj": self.board,
            "src": self.src,
            "dst": self.dst,
            "piece": self.piece,
            "killed": self.killed,
            "new_piece": self.new_piece,
        }

        return type(self)(**(defaults | params))

    def is_legal(self):
        test_obj = self.board + self
        return not test_obj[King, self.board.is_white_turn][0].in_check()


class BoardInfo:
    FENSYMBOLS: dict[str, Type[BasePiece]] = {
        "k": King,
        "q": Queen,
        "r": Rook,
        "b": Bishop,
        "n": Knight,
        "p": Pawn,
    }

    @classmethod
    def init_chess960(cls) -> "BoardInfo":
        pieces = [Pawn(BoardPoint(i, 1), None, True) for i in range(8)] + [
            Pawn(BoardPoint(i, 6), None, False) for i in range(8)
        ]
        free_columns = list(range(8))
        random.shuffle(free_columns)

        king = random.randint(1, 6)
        free_columns.remove(king)

        while True:
            rook1 = random.choice(free_columns)
            if king > rook1:
                free_columns.remove(rook1)
                break

        while True:
            rook2 = random.choice(free_columns)
            if rook2 > king:
                free_columns.remove(rook2)
                break

        bishop1 = random.choice(free_columns)
        free_columns.remove(bishop1)

        while True:
            bishop2 = random.choice(free_columns)
            if abs(bishop2 - bishop1) % 2:
                free_columns.remove(bishop2)
                break

        knight1, knight2, queen = free_columns

        for piece, column in zip(
            (King, Rook, Rook, Bishop, Bishop, Knight, Knight, Queen),
            (king, rook1, rook2, bishop1, bishop2, knight1, knight2, queen),
        ):
            pieces += [
                piece(BoardPoint(column, 0), None, True),
                piece(BoardPoint(column, 7), None, False),
            ]
        pieces[16].queenside_rook_pos = pieces[18].pos
        pieces[17].queenside_rook_pos = pieces[19].pos

        pieces[16].kingside_rook_pos = pieces[20].pos
        pieces[17].kingside_rook_pos = pieces[21].pos
        return cls(pieces, is_chess960=True)

    @classmethod
    def from_fen(cls, fen: str) -> "BoardInfo":
        pieces = []
        (
            board,
            is_white_turn,
            _castlings,
            enpassant_pos,
            empty_halfturns,
            turn,
        ) = fen.split(" ")

        is_chess960 = (
            not any([char in "KQkq" for char in _castlings])
            if _castlings != "-"
            else False
        )

        castlings = [[], []]
        if not is_chess960:
            _castlings = _castlings.translate(
                {
                    ord("K"): ord("H"),
                    ord("k"): ord("h"),
                    ord("q"): ord("a"),
                    ord("Q"): ord("A"),
                }
            )

        if _castlings != "-":
            for char in _castlings:
                castlings[char.isupper()].append(
                    BoardPoint(char.lower() + ("1" if char.isupper() else "8"))
                )
            castlings = [sorted(i, key=lambda x: x.file) for i in castlings]

        board = board.split("/")
        for line in range(8):
            offset = 0
            for column in range(8):
                if column + offset > 7:
                    break
                char = board[line][column]
                if char.isdigit():
                    offset += int(char) - 1
                else:
                    piece = cls.FENSYMBOLS[char.lower()](
                        BoardPoint(column + offset, 7 - line), None, char.isupper()
                    )
                    if isinstance(piece, King):
                        if (
                            len(castlings[piece.is_white]) == 2
                        ):  # TBD: `match` statement
                            (
                                piece.queenside_rook_pos,
                                piece.kingside_rook_pos,
                            ) = castlings[piece.is_white]
                        elif len(castlings[piece.is_white]) == 1:
                            if castlings[piece.is_white][0].file < piece.file:
                                piece.queenside_rook_pos = castlings[piece.is_white][0]
                            else:
                                piece.kingside_rook_pos = castlings[piece.is_white][0]

                    pieces.append(piece)

        return cls(
            pieces,
            is_white_turn=is_white_turn == "w",
            enpassant_pos=BoardPoint(enpassant_pos),
            empty_halfturns=int(empty_halfturns),
            turn=int(turn),
            is_chess960=is_chess960,
        )

    def __init__(
        self,
        board: list[BasePiece],
        is_white_turn: bool = True,
        enpassant_pos: BoardPoint = None,
        empty_halfturns: int = 0,
        turn: int = 1,
        is_chess960=False,
    ):
        self.is_chess960 = is_chess960
        self.board = board
        self.is_white_turn = is_white_turn
        if enpassant_pos:
            self.enpassant_pos = [
                BoardPoint(enpassant_pos.file, int((enpassant_pos.file + 4.5) // 2)),
                enpassant_pos,
            ]
        else:
            self.enpassant_pos = [
                None,
                None,
            ]  # #0 - actual position of the piece; #1 - position to move the pawn capturing en-passant to.
        self.empty_halfturns = empty_halfturns
        self.turn = turn

        for piece in board:
            piece.board = self

    def __repr__(self):
        return f"BoardInfo({self.get_fen()})"

    def __hash__(self):
        return hash(self._fen_board())

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
            return Move.from_piece(src_king, dst_king.pos)

        src_piece = sorted(
            src_pieces, key=lambda x: int(x.is_white == other.is_white_turn)
        )[-1]
        return Move.from_piece(
            src_piece,
            dst_pieces[0].pos,
            new_piece=type(dst_pieces[0])
            if type(src_piece) != type(dst_pieces[0])
            else None,
        )

    def __add__(self, move: Move) -> "BoardInfo":
        new = self.copy(
            turn=self.turn + (1 if not self.is_white_turn else 0),
            is_white_turn=not self.is_white_turn,
            empty_halfturns=0
            if move.is_capturing() or move.piece == Pawn
            else self.empty_halfturns + 1,
            enpassant_pos=move.enpassant_pos()[1],
        )
        piece = new[move.src]

        if move.is_capturing() and move.killed != King:
            del new[move.capture_dst]
            allied_king = new[King, new.is_white_turn][0]
            if move.dst == allied_king.queenside_rook_pos:
                allied_king.queenside_rook_pos = None
            elif move.dst == allied_king.kingside_rook_pos:
                allied_king.kingside_rook_pos = None

        if move.is_promotion():
            del new[move.src]
            promoted_piece = move.new_piece(piece.pos, self, self.is_white_turn)
            promoted_piece.promoted = True
            new.board.append(promoted_piece)
        if move.is_castling():
            new[move.rook_src] = move.rook_dst

        new[move.src] = move.dst

        return new

    def __getitem__(self, *keys) -> Union[BasePiece, list[BasePiece]]:
        keys = keys[0]
        if isinstance(keys, int):  # TBD: `match` statement
            return self.get_by_pos(keys, None)
        elif isinstance(keys, type):
            return self.get_by_type(keys, None)
        elif isinstance(keys, BoardPoint):
            return self.get_by_pos(keys.file, keys.rank)
        elif isinstance(keys, tuple):
            if type(keys[0]) == int:
                return self.get_by_pos(*keys)
            elif type(keys[0]) == type or type(keys[1]) == bool:
                return self.get_by_type(*keys)
            elif keys == (None, None):
                return self.board
        raise TypeError(f"Unknown argument: {keys}")

    def __setitem__(self, pos: BoardPoint, new_pos: BoardPoint):
        self[pos]._move(new_pos)

    def __delitem__(self, pos: BoardPoint):
        for index in range(len(self.board)):
            piece = self.board[index]
            if piece.pos == pos:
                if isinstance(piece, Rook):
                    allied_king = self[King, piece.is_white][0]
                    if pos == allied_king.kingside_rook_pos:
                        allied_king.kingside_rook_pos = None
                    elif pos == allied_king.queenside_rook_pos:
                        allied_king.queenside_rook_pos = None

                del self.board[index]
                return

        raise ValueError(f"piece to remove not found on {pos}.")

    def _fen_board(self) -> str:
        res = ""
        for row in range(7, -1, -1):
            for column in range(0, 8):
                piece = self[column, row]
                if piece:
                    res += piece.fen_symbol
                elif res and res[-1].isdigit():
                    res = res[:-1] + str(int(res[-1]) + 1)
                else:
                    res += "1"
            res += "/"

        return res[:-1]

    @property
    def whites(self) -> Iterator[BasePiece]:
        return filter(lambda x: x.is_white, self.board)

    @property
    def blacks(self) -> Iterator[BasePiece]:
        return filter(lambda x: not x.is_white, self.board)

    @property
    def castlings(self) -> str:
        res = ""
        white_king = self[King, True][0]
        black_king = self[King, False][0]
        if self.is_chess960:
            for is_white, pos in (
                (1, white_king.kingside_rook_pos),
                (1, white_king.queenside_rook_pos),
                (0, black_king.queenside_rook_pos),
                (0, black_king.kingside_rook_pos),
            ):
                res += chr(pos.file + 97 - 32 * is_white) if pos is not None else ""
        else:
            for char, pos in (
                ("K", white_king.kingside_rook_pos),
                ("Q", white_king.queenside_rook_pos),
                ("k", black_king.kingside_rook_pos),
                ("q", black_king.queenside_rook_pos),
            ):
                res += char if pos is not None else ""

        return res or "-"

    def get_fen(self) -> str:
        return " ".join(
            (
                self._fen_board(),
                "w" if self.is_white_turn else "b",
                self.castlings,
                str(self.enpassant_pos[1]) if self.enpassant_pos[0] else "-",
                str(self.empty_halfturns),
                str(self.turn),
            )
        )

    def get_cfen(self, version: int) -> bytes:  #   experiments
        res = f"{version:0>8b}{self.empty_halfturns:0>8b}{self.turn:0>8b}"
        white_king = self[King, True][0]
        black_king = self[King, False][0]
        castlings = (
            white_king.kingside_rook_pos,
            white_king.queenside_rook_pos,
            black_king.kingside_rook_pos,
            black_king.queenside_rook_pos,
        )

        if version == 1:
            res += str(int(self.is_white_turn))
            for piece_type in (Pawn, Knight, Bishop, Rook, Queen, King):
                pieces = self[piece_type]
                res += bin(len(pieces))[2:]
                if piece_type == Rook:  #    TBD: `match` statement
                    res += "".join(
                        f"{int(piece.is_white)}{int(piece.pos in castlings)}{int(piece.pos):0>6b}"
                        for piece in pieces
                    )
                elif piece_type == Pawn:
                    res += "".join(
                        f"{int(piece.is_white)}{int(piece.pos == self.enpassant_pos[0])}{int(piece.pos):0>6b}"
                        for piece in pieces
                    )
                else:
                    res += "".join(
                        f"{int(piece.is_white)}{int(piece.pos):0<6b}"
                        for piece in pieces
                    )
        elif version == 2:
            PIECE_TYPES = (Pawn, Knight, Bishop, Rook, Queen, King)
            ENPASSANT_CODE, NONE_CODE = (
                ("0000", "0001") if self.is_white_turn else ("0001", "0000")
            )
            for pos in range(64):
                pos = BoardPoint(pos)
                piece = self[pos]
                if pos == self.enpassant_pos[1]:  #   TBD: `match` statement
                    res += ENPASSANT_CODE
                elif piece is None:
                    res += NONE_CODE
                else:
                    piece_index = PIECE_TYPES.index(type(piece)) + 2
                    piece_index += (
                        1 if piece_index == 3 and piece.pos in castlings else 0
                    )
                    res += f"{piece_index + (7 if piece.is_white else 0):0>4b}"
        else:
            raise AssertionError("unknown CFEN version")

        n_bytes = len(res) / 8
        n_bytes = int(n_bytes) + (0 if int(n_bytes) == n_bytes else 1)
        return bytes([int(res[i * 8 : i * 8 + 8], 2) for i in range(n_bytes)])

    def debug(self) -> None:
        res = ""
        for line in range(7, -1, -1):
            res += f"     +---+---+---+---+---+---+---+---+\n{line + 1}    "
            for column in range(8):
                piece = self[column, line]
                res += f"| {piece.fen_symbol if piece else ' '} "
            res += "|\n"
        res += "     +---+---+---+---+---+---+---+---+\n       a   b   c   d   e   f   g   h"

        print(res)

    def copy(self, **new_params):
        params = {
            "is_white_turn": self.is_white_turn,
            "enpassant_pos": self.enpassant_pos[1],
            "empty_halfturns": self.empty_halfturns,
            "turn": self.turn,
        }
        return type(self)(
            [piece.copy() for piece in self.board], **(params | new_params)
        )

    def get_by_pos(self, column, row):
        res = []
        for piece in self.board:
            if (piece.pos.file == column or column is None) and (
                piece.pos.rank == row or row is None
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

    def get_taken_pieces(self, is_white) -> dict[Type["BasePiece"], int]:
        return {
            Pawn: 8
            - len(self[Pawn, is_white])
            + len(
                list(
                    filter(lambda x: x.promoted and x.is_white == is_white, self.board)
                )
            ),
            Knight: max(0, 2 - len(self[Knight, is_white])),
            Bishop: max(0, 2 - len(self[Bishop, is_white])),
            Rook: max(0, 2 - len(self[Rook, is_white])),
            Queen: max(0, 1 - len(self[Queen, is_white])),
        }
