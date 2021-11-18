import re
from typing import Iterable, Optional, TypedDict, cast
from .core import BoardInfo, Move, GameState
from .utils import _reversed, STARTPOS
from .base import get_dispatcher

MatchData = TypedDict(
    "MatchData",
    {
        "headers": dict[str, str],
        "white_name": str,
        "black_name": str,
        "result": GameState,
        "date": Optional[str],
        "moves": list[Move]
    },
)


class PGNParser:
    RESULT_CODES = {
        GameState.NORMAL: "*",
        GameState.CHECK: "*",
        GameState.ABORTED: "*",
        GameState.WHITE_CHECKMATED: "0-1",
        GameState.BLACK_CHECKMATED: "1-0",
        GameState.WHITE_RESIGNED: "0-1",
        GameState.BLACK_RESIGNED: "1-0",
        GameState.DRAW: "1/2-1/2",
        GameState.FIFTY_MOVE_DRAW: "1/2-1/2",
        GameState.INSUFFICIENT_MATERIAL_DRAW: "1/2-1/2",
        GameState.STALEMATE_DRAW: "1/2-1/2",
        GameState.THREEFOLD_REPETITION_DRAW: "1/2-1/2",
    }

    @classmethod
    def decode_moveseq(
        cls, src: str, startpos: BoardInfo = BoardInfo.from_fen(STARTPOS)
    ) -> tuple[list[Move], GameState]:
        moves: list[Move] = []
        *_moves, _result = src.replace("\n", " ").split()

        if _result in ("1/2-1/2", ".5-.5", "0.5-0.5"):  #   TBD: `match` statement
            result = GameState.DRAW
        if "#" in _moves[-1] or "++" in _moves[-1]:
            result = (
                GameState.BLACK_CHECKMATED
                if _result == "1-0"
                else GameState.WHITE_CHECKMATED
            )
        elif _result == "1-0":
            result = GameState.BLACK_RESIGNED
        elif _result == "0-1":
            result = GameState.WHITE_RESIGNED
        elif _result == "*":
            result = GameState.CHECK if "+" in _moves[-1] else GameState.NORMAL
        else:
            result = GameState.CHECK if "+" in _result else GameState.NORMAL
            _moves.append(_result)

        for token in _moves:
            if not (token[:-1].isdigit() and token[-1] == "."):
                moves.append(
                    Move.from_pgn(token, moves[-1].apply() if moves else startpos)
                )

        return moves, result

    @classmethod
    def encode_moveseq(
        cls,
        moves: list["Move"],
        result: Optional[GameState] = GameState.NORMAL,
        language_code: str = "en",
        line_length=80,
        turns_per_line=None,
    ) -> str:
        res = []

        for index, move in enumerate(moves):
            if index % 2 == 0:
                res.append(f"{move.board.turn}.")
            res.append(move.pgn_encode(language_code=language_code))

        if line_length:
            encoded = ""
            cur_line = ""
            for token in res + (
                [cls.RESULT_CODES[result]] if result is not None else []
            ):
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
    def decode(cls, src: str) -> MatchData:
        raw_headers, _moves = src.split("\n\n")
        _res = {}

        for header in raw_headers.splitlines():
            _key, _value = header.split(" ", maxsplit=1)
            _res[_key[1:]] = _value[1:-2]

        moves, result = cls.decode_moveseq(
            _moves, startpos=BoardInfo.from_fen(_res.get("FEN", STARTPOS))
        )
        res = MatchData(
            white_name=_res["White"], 
            black_name=_res["black"], 
            date=_res["date"],
            moves=moves,
            result=result,
            headers=_res
        )

        for key in ["FEN", "Result", "Date", "White", "Black"]:
            if key in res["headers"]:
                del res["headers"][key]

        return res

    @classmethod
    def encode(
        cls,
        moves: list[Move],
        white_name: str = None,
        black_name: str = None,
        date: str = None,
        result: GameState = GameState.NORMAL,
        headers: dict[str, str] = {},
    ):
        std_headers = {
            "Event": "Online Chess on Telegram",
            "Site": get_dispatcher().bot.link,
            "Date": date or "?",
            "Round": "-",
            "White": white_name or "?",
            "Black": black_name or "?",
            "Result": cls.RESULT_CODES[result],
        }
        startpos = moves[0].board.get_fen()
        if startpos != STARTPOS:
            std_headers["FEN"] = startpos
        std_headers.update(headers)

        return "\n".join(
            [
                "\n".join([f'[{k} "{v}"]' for k, v in std_headers.items()]),
                "",
                cls.encode_moveseq(moves, result=result),
            ]
        )


class CGNParser:
    RESULT_CODES = {
        GameState.NORMAL: b"*",
        GameState.CHECK: b"+",
        GameState.WHITE_CHECKMATED: b"WC",
        GameState.BLACK_CHECKMATED: b"BC",
        GameState.WHITE_RESIGNED: b"WR",
        GameState.BLACK_RESIGNED: b"BR",
        GameState.DRAW: b"AD",
        GameState.FIFTY_MOVE_DRAW: b"FMD",
        GameState.INSUFFICIENT_MATERIAL_DRAW: b"IMD",
        GameState.STALEMATE_DRAW: b"SD",
        GameState.THREEFOLD_REPETITION_DRAW: b"TRD",
    }
    SPECIAL_CHARS = {"##": "#", "#N": "\n", "#T": "\t"}

    @classmethod
    def escape(cls, src: str) -> str:
        return src.replace("#", "##").replace("\n", "#N").replace("\t", "#T")

    @classmethod
    def unescape(cls, src: str) -> str:
        return re.sub("#[#NT]", lambda match: cls.SPECIAL_CHARS[match[0]], src)

    @classmethod
    def decode(cls, src: bytes) -> MatchData:
        data: dict[bytes, bytes] = dict(cast(tuple[bytes, bytes], x.split(b"\t")) for x in src.splitlines())
        res = MatchData(
            white_name=data[b"W"].decode(),
            black_name=data[b"B"].decode(),
            date=data[b"D"].decode() if b"D" in data else None,
            result=_reversed(cls.RESULT_CODES)[data.get(b"R", b"*")],
            moves=[],
            headers={}
        )

        last_pos = BoardInfo.from_fen(
            data.get(b"S", STARTPOS.encode()).decode()
        )
        for move in zip(data[b"M"][::2], data[b"M"][1::2]):
            res["moves"].append(
                Move.from_hash(
                    int.from_bytes(b"%c%c" % move, byteorder="big"), last_pos
                )
            )
            last_pos = res["moves"][-1].apply()

        for k in [b"W", b"R", b"B", b"D", b"S", b"M"]:
            if k in data:
                del data[k]
        res["headers"] = {k.decode(): cls.unescape(v.decode()) for k, v in data.items()}

        return res

    @classmethod
    def encode(
        cls,
        moves: Iterable[Move],
        white_name: str = None,
        black_name: str = None,
        date: str = None,
        result: GameState = GameState.NORMAL,
        headers: dict[str, str] = {},
    ) -> bytes:
        res = {
            b"M": b"",
            b"W": (white_name or "?").encode(),
            b"B": (black_name or "?").encode(),
            b"R": cls.RESULT_CODES[result],
            b"D": (date or "?").encode()
        }
        res.update({k.encode(): cls.escape(v).encode() for k, v in headers.items()})

        for move in moves:
            res[b"M"] += hash(move).to_bytes(2, byteorder="big")

        return b"\n".join([k + b"\t" + v for k, v in res.items()])
