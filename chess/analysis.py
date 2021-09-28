import subprocess
import random
import itertools
import functools
from typing import Union
from .utils import decode_pos, encode_pos, BoardPoint
from .core import Move, BoardInfo, BasePiece, King

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
        board[decode_pos(raw[:2])],
        decode_pos(raw[2:4]),
        new_piece=raw[4] if len(raw) == 5 else "",
    )


def encode_engine_move(move: Move) -> str:
    res = encode_pos(move.src) + encode_pos(move.dst)
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
            pos = BoardPoint(column=file, row=rank)
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


def _parse_response(src: str) -> dict:
    formatted = (
        src.removeprefix("info ").replace("cp ", "cp#").replace("mate ", "mate#")
    )
    tokens = formatted.split(" ")
    res = dict(
        zip(tokens[: tokens.index("pv") : 2], tokens[1 : tokens.index("pv") : 2])
    )

    res["pv"] = tokens[tokens.index("pv") + 1 :]
    if "cp" in res["score"]:
        res["score"] = int(res["score"][3:]) / 100
    elif "mate" in res["score"]:
        res["score"] = (
            ("-" if "-" in res["score"] else "")
            + "M"
            + res["score"][5:].replace("-", "")
        )

    for key in res:
        if type(res[key]) == str and res[key].isdigit():
            res[key] = int(res[key])
    return res


class ChessEngine:
    def __init__(self, path):
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
        self.move_probabilities: tuple[int] = (0,)
        self.options = {}

    def __hash__(self):
        return self._process.pid

    def __setitem__(self, key, value):
        self.options[key] = value
        self._input(
            f"setoption name {key} value {str(value).lower() if isinstance(value, bool) else value}\n"
        )

    def _input(self, cmd: str) -> None:
        self._process.stdin.write(cmd)
        # print("Process <-", repr(cmd))

    def set_move_probabilities(self, values: tuple[int]) -> None:
        self.move_probabilities = values
        self["MultiPV"] = max(values) + 1

    def get_moves(
        self, board: BoardInfo, depth: int, **kwargs
    ) -> list[dict[str, Union[str, int, float, list[Move]]]]:
        self._input(
            f"position fen {board.fen}\ngo depth {depth} { ' '.join([f'{k} {v}' for k, v in kwargs.items()]) }\n"
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
                parsed = _parse_response(line[:-1])
                if (
                    parsed["multipv"] not in [i["multipv"] for i in results]
                    and parsed["depth"] == depth
                ):
                    if not board.is_white_turn:
                        if isinstance(parsed["score"], str):
                            parsed["score"] = (
                                parsed["score"].replace("-", "")
                                if "-" in parsed["score"]
                                else "-" + parsed["score"]
                            )
                        else:
                            parsed["score"] *= -1
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

    def eval_position_static(self, board: BoardInfo) -> float:
        self._process.stdin.write(f"position fen {board.fen}\neval\n")
        for line in self._process.stdout:
            if "Final evaluation" in line:
                return float(line.split()[2])

    @functools.lru_cache(maxsize=32)
    def eval_position(self, move: Move, depth: int) -> Union[float, str]:
        res = self.get_moves(move.board, depth, searchmoves=encode_engine_move(move))
        return res[0]["score"]

    def eval_move(self, move: Move, depth: int, prev_move: Move = None) -> str:
        n_moves = 0
        for piece in move.board.whites if move.is_white else move.board.blacks:
            possible_moves = [move for move in piece.get_moves() if move.is_legal()]
            n_moves += len(possible_moves)
        if n_moves == 1:
            return "□"

        if move == self.get_moves(move.board, depth)[0]["pv"][0]:
            return "!!!"

        if prev_move:
            prev_eval = self.eval_position(prev_move, depth)
        else:
            prev_eval = self.eval_position_static(move.board)
        cur_eval = self.eval_position(move, depth)

        if type(prev_eval) == str and type(cur_eval) == str:
            if int(prev_eval[-1]) > int(cur_eval[-1]):
                return "!!" if "-" in prev_eval == move.is_white else "??"
            elif int(prev_eval[-1]) < int(cur_eval[-1]):
                return "??" if "-" in prev_eval == move.is_white else "!!"
            else:
                return "!"
        elif type(prev_eval) == str:
            if "-" in prev_eval == move.is_white:
                return "!!"
            else:
                return "??"
        elif type(cur_eval) == str:
            if "-" in cur_eval != move.is_white:
                return "??"
            else:
                return "!!"

        eval_coef = (cur_eval - prev_eval) * (1 if move.is_white else -1)
        if eval_coef >= 1.0:
            return "!!"
        elif 0.0 <= eval_coef < 1.0:
            return "!"
        elif -1.0 <= eval_coef < 0.0:
            return "?!"
        elif -2.0 <= eval_coef < -1.0:
            return "?"
        else:
            return "??"
