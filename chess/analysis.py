import subprocess
import random
import functools
from typing import Optional, Sequence, Union, TypedDict
from .utils import BoardPoint
from .core import Move, BoardInfo, MoveEval

EngineResponse = TypedDict("EngineResponse", {"moves": list[Move], "depth": int, "multipv": int, "score": "EvalScore"})


def decode_engine_move(raw: str, board: BoardInfo) -> Move:
    piece = board[BoardPoint(raw[:2])]
    assert piece is not None, f"{raw}"
    return Move.from_piece(
        piece,
        BoardPoint(raw[2:4]),
        new_piece=BoardInfo.FENSYMBOLS[raw[4]] if len(raw) == 5 else None,
    )


def encode_engine_move(move: Move) -> str:
    res = str(move.src) + str(move.dst)
    return res + (move.new_piece.fen_symbols[0] if move.new_piece else "")


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
        (-1, -1): False,
    }

    def __init__(self, arg: Union[int ,float], is_white_side: bool):
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

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, EvalScore):
            return NotImplemented
        return self.score == other.score and self.mate_in == other.mate_in

    def __compare(self, other: "EvalScore", side: bool):
        if self.__class__ != other.__class__:
            return NotImplemented

        comp = (
            round((self.score - other.score) / abs(self.score - other.score or 1)),
            (self.mate_in - other.mate_in) // abs(self.mate_in - other.mate_in or 1),
        )

        res = self.comparison_map[comp]
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
        self._process.stdout.readline()   # type: ignore
        self.default_eval_depth = default_eval_depth
        self.move_probabilities: Sequence[int] = (0,)
        self.options: dict[str, Union[str, int]] = {}
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
    def _parse_response(src: str, board: BoardInfo, depth: int) -> Optional[EngineResponse]:
        tokens = src.removeprefix("info ").replace("cp ", "cp#").replace("mate ", "mate#").split(" ")
        _res = dict(zip(tokens[: tokens.index("pv") : 2], tokens[1 : tokens.index("pv") : 2]))
        eval_type, eval_score = _res["score"].split("#")

        res = EngineResponse(
            depth=int(_res["depth"]), 
            multipv=int(_res["multipv"]), 
            score=EvalScore(
                int(eval_score) / 100 if eval_type == "cp" else int(eval_score), board.is_white_turn
            ),
            moves=[]
        )
        if res["depth"] != depth:
            return None

        res["score"].score *= 1 if board.is_white_turn else -1
        res["score"].mate_in *= 1 if board.is_white_turn else -1

        _moves = tokens[tokens.index("pv") + 1 :]
        for index in range(len(_moves)):
            res["moves"].append(decode_engine_move(_moves[index], board))
            board = res["moves"][index].apply()

        return res

    def _input(self, cmd: str) -> None:
        self._process.stdin.write(cmd)   # type: ignore
        # print("Process <-", repr(cmd))

    def set_move_probabilities(self, values: list[int]) -> None:
        self.move_probabilities = values

    def get_moves(
        self, board: BoardInfo, depth: int = None, **kwargs
    ) -> list[EngineResponse]:
        if depth is None:
            depth = self.default_eval_depth
        assert depth is not None
        self._input(
            f"position fen {board.get_fen()}\ngo depth {depth} { ' '.join([f'{k} {v}' for k, v in kwargs.items()]) }\n"
        )
        self._process.stdout.readline()   # type: ignore
        results: list[EngineResponse] = []
        for line in self._process.stdout:   # type: ignore
            # print("Process ->", repr(line))
            if "bestmove" in line:
                break
            elif "currmove" in line:
                continue
            elif " depth" in line:
                parsed = self._parse_response(line[:-1], board, depth)
                if parsed is not None:
                    results.append(parsed)

        return results

    def get_move(self, *args, **kwargs) -> Move:
        res = self.get_moves(*args, **kwargs)
        return res[min(len(res) - 1, random.choice(self.move_probabilities))]["moves"][0]

    def eval_position_static(self, board: BoardInfo) -> EvalScore:
        self._input(f"position fen {board.get_fen()}\neval\n")
        for line in self._process.stdout:   #  type: ignore
            if "Final evaluation" in line:
                return EvalScore(float(line.split()[2]), board.is_white_turn)

        assert False

    @functools.lru_cache(maxsize=32)
    def eval_position(self, move: Move, depth: int = None) -> EvalScore:
        res = self.get_moves(move.board, depth, searchmoves=encode_engine_move(move))
        return res[0]["score"]

    def eval_move(
        self, move: Move, depth: int = None, prev_move: Move = None
    ) -> None:
        n_moves = 0
        cur_eval = self.eval_position(move, depth)
        for piece in move.board.whites if move.is_white else move.board.blacks:
            n_moves += len(piece.get_moves())
        if n_moves == 1:
            move.metadata["pos_eval"] = cur_eval
            move.metadata["move_eval"] = MoveEval.FORCED
            move.metadata["best_move"] = move
            move.metadata["best_move_eval"] = cur_eval

        if prev_move:
            prev_eval = self.eval_position(prev_move, depth)
        else:
            prev_eval = self.eval_position_static(move.board)

        best_line, second_best_line, _ = self.get_moves(move.board, depth)
        if move == best_line["moves"][0]:
            mark = (
                MoveEval.PRECISE
                if second_best_line["score"] < prev_eval
                else MoveEval.BEST
            )

        else:
            comp = (
                round(
                    (cur_eval.score - prev_eval.score)
                    / abs(cur_eval.score - prev_eval.score or 1)
                ),
                (cur_eval.mate_in - prev_eval.mate_in)
                // abs(cur_eval.mate_in - prev_eval.mate_in or 1),
            )
            mark = MoveEval.GOOD

            if comp[0] == comp[1] != 0:  #    (1, 1) or (-1, -1)
                mark = (
                    MoveEval.GREAT
                    if cur_eval.is_white_side == (comp[0] > 0)
                    else MoveEval.BLUNDER
                )
            elif comp == (0, 1):
                if cur_eval.is_white_side:
                    if prev_eval.mate_in < 0 and cur_eval.mate_in > 0:
                        mark = MoveEval.GREAT
                    elif (
                        prev_eval.mate_in + cur_eval.mate_in < 0
                        and abs(prev_eval.mate_in - cur_eval.mate_in) >= 5
                    ):
                        mark = MoveEval.BLUNDER
                    else:
                        mark = MoveEval.MISTAKE
                else:
                    if prev_eval.mate_in < 0 and cur_eval.mate_in > 0:
                        mark = MoveEval.BLUNDER
                    else:
                        mark = MoveEval.GREAT
            elif comp[0] * comp[1] == -1:  #     (-1, 1) or (1, -1)
                if (comp[0] > 0) != cur_eval.is_white_side:
                    mark = MoveEval.GREAT
                elif cur_eval.mate_in != 0:
                    mark = MoveEval.BLUNDER
                else:
                    mark = MoveEval.MISTAKE
            elif comp[0] != 0 and comp[1] == 0:  #   (-1, 0) or (1, 0)
                if abs(prev_eval.score - cur_eval.score) > 0.5:
                    mark = (
                        MoveEval.GREAT
                        if cur_eval.is_white_side == prev_eval.score > cur_eval.score
                        else MoveEval.MISTAKE
                    )
                else:
                    mark = (
                        MoveEval.GOOD
                        if cur_eval.is_white_side == prev_eval.score > cur_eval.score
                        else MoveEval.WEAK
                    )
            elif comp == (0, -1):
                if not cur_eval.is_white_side:
                    if prev_eval.mate_in > 0 and cur_eval.mate_in < 0:
                        mark = MoveEval.GREAT
                    elif (
                        prev_eval.mate_in + cur_eval.mate_in < 0
                        and abs(prev_eval.mate_in - cur_eval.mate_in) >= 5
                    ):
                        mark = MoveEval.BLUNDER
                    else:
                        mark = MoveEval.MISTAKE
                else:
                    if prev_eval.mate_in > 0 and cur_eval.mate_in < 0:
                        mark = MoveEval.BLUNDER
                    else:
                        mark = MoveEval.GREAT

        move.metadata["pos_eval"] = cur_eval
        move.metadata["move_eval"] = mark
        move.metadata["best_move"] = best_line["moves"][0]
        move.metadata["best_move_eval"] = best_line["score"]
