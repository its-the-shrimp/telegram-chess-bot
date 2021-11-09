import subprocess
import random
import itertools
import functools
from typing import Union, overload
from .utils import BoardPoint
from .core import Move, BoardInfo, King, MoveEval

IDEAL_N_CTRL_POS = {
    "Pawn": 2,
    "Knight": 8,
    "Bishop": 14,
    "Rook": 14,
    "Queen": 28,
    "King": 8,
}


def decode_engine_move(raw: str, board: BoardInfo) -> Move:
    return Move.from_piece(
        board[BoardPoint(raw[:2])],
        BoardPoint(raw[2:4]),
        new_piece=BoardInfo.FENSYMBOLS[raw[4]] if len(raw) == 5 else None,
    )


def encode_engine_move(move: Move) -> str:
    res = str(move.src) + str(move.dst)
    return res + (move.new_piece.fen_symbol[0] if move.new_piece else "")


def eval_piece_activity(board: BoardInfo) -> float:
    white_value = 0.0
    black_value = 0.0
    all_white_moves = tuple(
        itertools.chain(
            *[
                map(
                    lambda x: x.dst,
                    filter(
                        Move.is_legal,
                        piece.get_moves(all=True, only_capturing=True),
                    ),
                )
                for piece in board.whites
            ]
        )
    )
    all_black_moves = tuple(
        itertools.chain(
            *[
                map(
                    lambda x: x.dst,
                    filter(
                        Move.is_legal,
                        piece.get_moves(all=True, only_capturing=True),
                    ),
                )
                for piece in board.blacks
            ]
        )
    )

    for piece in board.board:
        if type(piece) == King:
            continue
        n_controlled_pos = 0
        for move in filter(
            Move.is_legal, piece.get_moves(all=True, only_capturing=True)
        ):
            enemy_moves = all_black_moves if piece.is_white else all_white_moves
            existing_piece = board[move.dst]
            if existing_piece:
                if move.dst in enemy_moves:
                    n_controlled_pos += 2
                elif existing_piece.is_white != piece.is_white:
                    n_controlled_pos += 5 - enemy_moves.count(move.dst)

            else:
                n_controlled_pos += 1

        if piece.is_white:
            white_value += piece.value * (
                n_controlled_pos / IDEAL_N_CTRL_POS[type(piece).__name__]
            )
        else:
            black_value += piece.value * (
                n_controlled_pos / IDEAL_N_CTRL_POS[type(piece).__name__]
            )

    res = white_value - black_value
    if res > 0:
        if black_value:
            return res / black_value * 100
        return 100.0
    elif res < 0:
        if white_value:
            return res / white_value * 100
        return -100.0
    else:
        return 0.0


def eval_pieces_defense(board: BoardInfo) -> dict[BoardPoint, float]:
    all_white_moves = tuple(
        itertools.chain(
            *[
                filter(
                    Move.is_legal,
                    piece.get_moves(all=True, only_capturing=True),
                )
                for piece in board.whites
            ]
        )
    )
    all_black_moves = tuple(
        itertools.chain(
            *[
                filter(
                    Move.is_legal,
                    piece.get_moves(all=True, only_capturing=True),
                )
                for piece in board.blacks
            ]
        )
    )
    whites_value_sum = len([piece.value for piece in board.whites])
    blacks_value_sum = sum([piece.value for piece in board.blacks])

    res = {}
    for file in range(8):
        for rank in range(8):
            pos = BoardPoint(file, rank)
            if board[pos]:
                res[pos] = len([i.piece.value for i in all_white_moves if i.dst == pos])
                res[pos] -= len(
                    [i.piece.value for i in all_black_moves if i.dst == pos]
                )

                if res[pos] > 0:
                    res[pos] /= blacks_value_sum
                elif res[pos] < 0:
                    res[pos] /= whites_value_sum
                else:
                    res[pos] = 0.0

    return res


def eval_position_exp(board: BoardInfo) -> float:
    defense_score = tuple(eval_pieces_defense(board).values())
    defense_score = sum(defense_score) / len(defense_score)
    activity_score = eval_piece_activity(board)
    print("activity", activity_score, "defense", defense_score)

    return round((defense_score + activity_score) / 2, 2)


class EvalScore:
    comparison_map = {
        (1, 1): True,
        (1, 0): True,
        (1, -1): False,
        (0, 1): False,
        (0, 0): None,
        (0, -1): True,
        (-1, 1): True,
        (-1, 0): False,
        (-1, -1): False
    }

    @overload
    def __init__(self, eval_score: float, is_white_side: bool):
        ...

    @overload
    def __init__(self, mate_in: int, is_white_side: bool):
        ...

    def __init__(self, arg, is_white_side: bool):
        if isinstance(arg, int):
            self.score = 0.0
            self.mate_in = arg
        elif isinstance(arg, float):
            self.score = arg
            self.mate_in = 0

        self.is_white_side = is_white_side

    def __str__(self):
        if self.mate_in != 0:
            return ("+" if self.mate_in > 0 else "-") + f"M{abs(self.mate_in)}"
        else:
            return ("+" if self.score > 0 else "") + f"{self.score:.2f}"

    def __repr__(self):
        return f"<EvalScore({self.__str__()})>"

    def __eq__(self, other: "EvalScore"):
        if self.__class__ != other.__class__:
            return NotImplemented
        return self.score == other.score and self.mate_in == other.mate_in

    def __compare(self, other: "EvalScore", side: bool):
        if self.__class__ != other.__class__:
            return NotImplemented
        
        comp = (
            round((self.score - other.score) / abs(self.score - other.score or 1)),
            (self.mate_in - other.mate_in) // abs(self.mate_in - other.mate_in or 1)
        )

        res = self.comparison_map[tuple(comp)]
        return res == side if res is not None else False

    def __gt__(self, other: "EvalScore"):
        return self.__compare(other, self.is_white_side)

    def __lt__(self, other: "EvalScore"):
        return self.__compare(other, not self.is_white_side)


class ChessEngine:
    def __init__(self, path, default_eval_depth: int = None):
        self._process = subprocess.Popen(
            path,
            bufsize=1,
            universal_newlines=True,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._process.stdout.readline()
        self.default_eval_depth = default_eval_depth
        self.move_probabilities: tuple[int] = (0,)
        self.options = {}
        self["multiPV"] = 3

    def __hash__(self):
        return self._process.pid

    def __setitem__(self, key: str, value: Union[str, int, bool]):
        self.options[key] = value
        self._input(
            f"setoption name {key} value {str(value).lower() if isinstance(value, bool) else value}\n"
        )

    def __getitem__(self, key: str):
        return self.options[key]

    @staticmethod
    def _parse_response(src: str, is_white: bool) -> dict:
        formatted = (
            src.removeprefix("info ").replace("cp ", "cp#").replace("mate ", "mate#")
        )
        tokens = formatted.split(" ")
        res = dict(
            zip(tokens[: tokens.index("pv") : 2], tokens[1 : tokens.index("pv") : 2])
        )
        res["pv"] = tokens[tokens.index("pv") + 1 :]

        for key in res:
            if isinstance(res[key], str) and res[key].isdigit():
                res[key] = int(res[key])

        eval_type, eval_score = res["score"].split("#")
        res["score"] = EvalScore(int(eval_score) / 100 if eval_type=="cp" else int(eval_score), is_white)
        res["score"].score *= 1 if is_white else -1
        res["score"].mate_in *= 1 if is_white else -1

        return res

    def _input(self, cmd: str) -> None:
        self._process.stdin.write(cmd)
        # print("Process <-", repr(cmd))

    def set_move_probabilities(self, values: tuple[int]) -> None:
        self.move_probabilities = values

    def get_moves(
        self, board: BoardInfo, depth: int = None, **kwargs
    ) -> list[dict[str, Union[str, int, float, list[Move]]]]:
        depth = depth or self.default_eval_depth
        self._input(
            f"position fen {board.get_fen()}\ngo depth {depth} { ' '.join([f'{k} {v}' for k, v in kwargs.items()]) }\n"
        )
        self._process.stdout.readline()
        results = []
        for line in self._process.stdout:
            # print("Process ->", repr(line))
            if "bestmove" in line:
                break
            elif "currmove" in line:
                continue
            elif " depth" in line:
                parsed = self._parse_response(line[:-1], board.is_white_turn)
                if (
                    parsed["multipv"] not in [i["multipv"] for i in results]
                    and parsed["depth"] == depth
                ):
                    results.append(parsed)

        filtered = [i for i in results if i["depth"] == depth]
        if not filtered:
            filtered = results

        for line in filtered:
            for index in range(len(line["pv"])):
                prev_board = (
                    line["pv"][index - 1].board + line["pv"][index - 1]
                    if index
                    else board
                )
                line["pv"][index] = decode_engine_move(line["pv"][index], prev_board)

        return filtered

    def get_move(self, *args, **kwargs) -> Move:
        res = self.get_moves(*args, **kwargs)
        return res[random.choice(self.move_probabilities)]["pv"][0]

    def eval_position_static(self, board: BoardInfo) -> EvalScore:
        self._process.stdin.write(f"position fen {board.get_fen()}\neval\n")
        for line in self._process.stdout:
            if "Final evaluation" in line:
                return EvalScore(float(line.split()[2]), board.is_white_turn)

    @functools.lru_cache(maxsize=32)
    def eval_position(self, move: Move, depth: int = None) -> EvalScore:
        res = self.get_moves(move.board, depth, searchmoves=encode_engine_move(move))
        return res[0]["score"]

    def eval_move(self, move: Move, depth: int = None, prev_move: Move = None) -> tuple[MoveEval, Move, EvalScore]:
        n_moves = 0
        for piece in move.board.whites if move.is_white else move.board.blacks:
            n_moves += len(piece.get_moves())
        if n_moves == 1:
            return (MoveEval.FORCED, move)

        if prev_move:
            prev_eval = self.eval_position(prev_move, depth)
        else:
            prev_eval = self.eval_position_static(move.board)
        cur_eval = self.eval_position(move, depth)

        best_line, second_best_line, _ = self.get_moves(move.board, depth)
        if move == best_line["pv"][0]:
            mark = MoveEval.PRECISE if second_best_line["score"] < prev_eval else MoveEval.BEST
        
        else:
            comp = (
                round((cur_eval.score - prev_eval.score) / abs(cur_eval.score - prev_eval.score or 1)),
                (cur_eval.mate_in - prev_eval.mate_in) // abs(cur_eval.mate_in - prev_eval.mate_in or 1)
            )
            mark = MoveEval.GOOD

            if comp[0] == comp[1] != 0:   #    (1, 1) or (-1, -1)
                mark = MoveEval.GREAT if cur_eval.is_white_side == (comp[0] > 0) else MoveEval.BLUNDER
            elif comp == (0, 1):
                if cur_eval.is_white_side:
                    if prev_eval.mate_in < 0 and cur_eval.mate_in > 0:
                        mark = MoveEval.GREAT
                    elif prev_eval.mate_in + cur_eval.mate_in < 0 and abs(prev_eval.mate_in - cur_eval.mate_in) >= 5:
                        mark = MoveEval.BLUNDER
                    else:
                        mark = MoveEval.MISTAKE
                else:
                    if prev_eval.mate_in < 0 and cur_eval.mate_in > 0:
                        mark = MoveEval.BLUNDER
                    else:
                        mark = MoveEval.GREAT
            elif comp[0] * comp[1] == -1: #     (-1, 1) or (1, -1)
                if (comp[0] > 0) != cur_eval.is_white_side:
                    mark = MoveEval.GREAT
                elif cur_eval.mate_in != 0:
                    mark = MoveEval.BLUNDER
                else:
                    mark = MoveEval.MISTAKE
            elif comp[0] != 0 and comp[1] == 0:  #   (-1, 0) or (1, 0)
                if abs(prev_eval.score - cur_eval.score) > 0.5:
                    mark = MoveEval.GREAT if cur_eval.is_white_side == prev_eval.score > cur_eval.score else MoveEval.MISTAKE
                else:
                    mark = MoveEval.GOOD if cur_eval.is_white_side == prev_eval.score > cur_eval.score else MoveEval.WEAK
            elif comp == (0, -1):
                if not cur_eval.is_white_side:
                    if prev_eval.mate_in > 0 and cur_eval.mate_in < 0:
                        mark = MoveEval.GREAT
                    elif prev_eval.mate_in + cur_eval.mate_in < 0 and abs(prev_eval.mate_in - cur_eval.mate_in) >= 5:
                        mark = MoveEval.BLUNDER
                    else:
                        mark = MoveEval.MISTAKE
                else:
                    if prev_eval.mate_in > 0 and cur_eval.mate_in < 0:
                        mark = MoveEval.BLUNDER
                    else:
                        mark = MoveEval.GREAT

        return mark, best_line["pv"][0], best_line["score"]
            
