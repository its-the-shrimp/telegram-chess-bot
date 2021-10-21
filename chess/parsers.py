from datetime import datetime
import re
from typing import Generator, Iterable
from .core import *
from .utils import DATE_FORMAT


DRAW = "1/2-1/2"
WHITE_WIN = "1-0"
BLACK_WIN = "0-1"
NOT_FINISHED = "*"


def get_moves(boards: list["BoardInfo"]) -> Generator["Move", None, None]:
    for index in range(1, len(boards)):
        yield boards[index] - boards[index - 1]


class PGNParser:
    @classmethod
    def decode_moveseq(
        cls, src: str, startpos: str = STARTPOS
    ) -> tuple[list["BoardInfo"], str]:
        states = [BoardInfo.from_fen(startpos)]
        *moves, result = src.replace("\n", " ").split()

        result = DRAW if result in [".5-.5", "0.5-0.5"] else result
        if result not in [NOT_FINISHED, DRAW, WHITE_WIN, BLACK_WIN]:
            moves.append(result)
            result = NOT_FINISHED

        for token in moves:
            if not (token[:-1].isdigit() and token[-1] == "."):
                states.append(states[-1] + Move.from_pgn(token, states[-1]))

        return states, result

    @classmethod
    def encode_moveseq(
        cls,
        positions: Union[Iterable["BoardInfo"], Iterable["Move"]],
        result: str = "*",
        language_code: str = "en",
        line_length=80,
        turns_per_line=None,
    ) -> str:
        res = []
        moves = positions if hasattr(positions, "__len__") else get_moves(positions)

        for index, move in enumerate(moves):
            if index % 2 == 0:
                res.append(f"{move.board.turn}.")
            res.append(move.pgn_encode(language_code=language_code))

        if line_length:
            encoded = ""
            cur_line = ""
            for token in res + ([result] if result is not None else []):
                if any(
                    [
                        line_length and len(cur_line + " " + token) > line_length,
                        turns_per_line and cur_line.count(" ") == 2 * turns_per_line,
                    ]
                ):
                    encoded += cur_line + "\n"
                    cur_line = ""
                cur_line += (" " if cur_line else "") + token
            encoded += cur_line

            return encoded
        else:
            return " ".join(res)

    @classmethod
    def decode(cls, src: str) -> dict:
        raw_headers, moves = src.split("\n\n")
        res = {"headers": {}}

        for header in raw_headers.splitlines():
            _key, _value = header.split(" ", maxsplit=1)
            res["headers"][_key[1:]] = _value[1:-2]
        res["white_name"] = res["headers"]["White"]
        res["black_name"] = res["headers"]["Black"]
        res["date"] = datetime.strptime(res["headers"]["Date"], DATE_FORMAT)
        del res["headers"]["Date"], res["headers"]["White"], res["headers"]["Black"]

        res["states"], res["result"] = cls.decode_moveseq(
            moves, startpos=res["headers"].get("FEN", STARTPOS)
        )
        for key in ["FEN", "Result"]:
            if key in res["headers"]:
                del res["headers"][key]
        return res

    @classmethod
    def encode(
        cls,
        states: Iterable["BoardInfo"] = [],
        white_name: str = "?",
        black_name: str = "?",
        date: str = None,
        result: str = NOT_FINISHED,
        headers: dict[str, str] = {},
    ):
        std_headers = {
            "Event": "Online Chess on Telegram",
            "Site": "t.me/real_chessbot",
            "Date": date or "?",
            "Round": "-",
            "White": white_name,
            "Black": black_name,
            "Result": result,
        }
        startpos = states[0].fen
        if startpos != STARTPOS:
            std_headers["FEN"] = startpos
        headers = std_headers | headers

        return "\n".join(
            [
                "\n".join([f'[{k} "{v}"]' for k, v in headers.items()]),
                "",
                cls.encode_moveseq(get_moves(states), result=result),
            ]
        )


class CGNParser:
    RESULT_CODES = {DRAW: b"D", WHITE_WIN: b"W", BLACK_WIN: b"B", NOT_FINISHED: b"*"}
    SPECIAL_CHARS = {"##": "#", "#N": "\n", "#T": "\t"}

    @classmethod
    def escape(cls, src: str) -> str:
        return src.replace("#", "##").replace("\n", "#N").replace("\t", "#T")

    @classmethod
    def unescape(cls, src: str) -> str:
        return re.sub("#[#NT]", lambda match: cls.SPECIAL_CHARS[match[0]], src)

    @classmethod
    def decode(cls, src: bytes) -> dict:
        data = {"headers": dict(x.split(b"\t") for x in src.splitlines())}
        data["white_name"] = data["headers"].get(b"W")
        data["black_name"] = data["headers"].get(b"B")
        data["date"] = (
            data["headers"][b"D"].decode() if b"D" in data["headers"] else None
        )
        data["result"] = {v: k for k, v in cls.RESULT_CODES}[
            data["headers"].get(b"R", NOT_FINISHED)
        ]

        boards = [
            BoardInfo.from_fen(data["headers"].get(b"S", STARTPOS.encode()).decode())
        ]
        for move in map(
            b"".join, zip(data["headers"][b"M"][::2], data["headers"][b"M"][1::2])
        ):
            boards.append(
                boards[-1]
                + Move.from_hash(int.from_bytes(move, byteorder="big"), boards[-1])
            )

        for k in b"WRBDSM":
            if chr(k) in data["headers"]:
                del data["headers"][chr(k)]

        data["headers"] = {k.decode(): cls.unescape(v.decode()) for k, v in data["headers"].items()}

        return data

    @classmethod
    def encode(
        cls,
        states: Iterable["BoardInfo"] = [],
        white_name: str = "?",
        black_name: str = "?",
        date: datetime = None,
        result: str = NOT_FINISHED,
        headers: dict[str, str] = {},
    ):
        res = {
            b"M": b"",
            b"W": white_name.encode(),
            b"B": black_name.encode(),
            b"R": cls.RESULT_CODES[result],
        }
        if date is not None:
            res[b"D"] = date.encode()
        res.update({k.encode(): cls.escape(v).encode() for k, v in headers.items()})

        for move in get_moves(states):
            res[b"M"] += hash(move).to_bytes(2, byteorder="big")

        return b"\n".join([k + b"\t" + v for k, v in res.items()])
