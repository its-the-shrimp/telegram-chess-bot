import re
from typing import Generator, Iterable, Optional
from .core import BoardInfo, Move, GameState
from .utils import _reversed, STARTPOS


def get_moves(boards: list["BoardInfo"]) -> Generator["Move", None, None]:
    for index in range(1, len(boards)):
        yield boards[index] - boards[index - 1]


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
        GameState.INSUFFIECENT_MATERIAL_DRAW: "1/2-1/2",
        GameState.STALEMATE_DRAW: "1/2-1/2",
        GameState.THREEFOLD_REPETITION_DRAW: "1/2-1/2",
    }

    @classmethod
    def decode_moveseq(
        cls, src: str, startpos: str = STARTPOS
    ) -> tuple[list["BoardInfo"], GameState]:
        boards = [BoardInfo.from_fen(startpos)]
        *_moves, _result = src.replace("\n", " ").split()

        for token in _moves:
            if not (token[:-1].isdigit() and token[-1] == "."):
                boards.append(boards[-1] + Move.from_pgn(token, boards[-1]))

        if _result in ("1/2-1/2", ".5-.5", "0.5-0.5"):
            result = GameState.DRAW
        if _result in ("1-0", "0-1"):
            if "#" in _moves[-1] or "++" in _moves[-1]:
                result = (
                    GameState.WHITE_CHECKMATED
                    if boards[-1].is_white_turn
                    else GameState.BLACK_CHECKMATED
                )
            else:
                result = (
                    GameState.WHITE_RESIGNED
                    if boards[-1].is_white_turn
                    else GameState.BLACK_RESIGNED
                )
        else:
            result = GameState.CHECK if "+" in _moves[-1] else GameState.NORMAL

        return boards, result

    @classmethod
    def encode_moveseq(
        cls,
        moves: list["Move"] = None,
        positions: Iterable["BoardInfo"] = None,
        result: Optional[GameState] = GameState.NORMAL,
        language_code: str = "en",
        line_length=80,
        turns_per_line=None,
    ) -> str:
        res = []
        if moves is None:
            moves = get_moves(positions)

        for index, move in enumerate(moves):
            if index % 2 == 0:
                res.append(f"{move.board.turn}.")
            res.append(move.pgn_encode(language_code=language_code))

        if line_length:
            encoded = ""
            cur_line = ""
            for token in res + ([cls.RESULT_CODES[result]] if result is not None else []):
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
        res["date"] = res["headers"]["Date"]
        del res["headers"]["Date"], res["headers"]["White"], res["headers"]["Black"]

        res["boards"], res["result"] = cls.decode_moveseq(
            moves, startpos=res["headers"].get("FEN", STARTPOS)
        )
        for key in ["FEN", "Result"]:
            if key in res["headers"]:
                del res["headers"][key]
        return res

    @classmethod
    def encode(
        cls,
        boards: Iterable["BoardInfo"] = [],
        white_name: str = "?",
        black_name: str = "?",
        date: str = "?",
        result: GameState = GameState.NORMAL,
        headers: dict[str, str] = {},
    ):
        std_headers = {
            "Event": "Online Chess on Telegram",
            "Site": "t.me/real_chessbot",
            "Date": date,
            "Round": "-",
            "White": white_name,
            "Black": black_name,
            "Result": result,
        }
        startpos = boards[0].get_fen()
        if startpos != STARTPOS:
            std_headers["FEN"] = startpos
        headers = std_headers | headers

        return "\n".join(
            [
                "\n".join([f'[{k} "{v}"]' for k, v in headers.items()]),
                "",
                cls.encode_moveseq(positions=boards, result=result),
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
        GameState.INSUFFIECENT_MATERIAL_DRAW: b"IMD",
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
    def decode(cls, src: bytes) -> dict:
        data = {"headers": dict(x.split(b"\t") for x in src.splitlines())}
        data["white_name"] = data["headers"].get(b"W").decode()
        data["black_name"] = data["headers"].get(b"B").decode()
        data["date"] = (
            data["headers"][b"D"].decode() if b"D" in data["headers"] else None
        )
        data["result"] = _reversed(cls.RESULT_CODES)[data["headers"].get(b"R", b"*")]

        data["boards"] = [
            BoardInfo.from_fen(data["headers"].get(b"S", STARTPOS.encode()).decode())
        ]
        for move in zip(data["headers"][b"M"][::2], data["headers"][b"M"][1::2]):
            data["boards"].append(
                data["boards"][-1]
                + Move.from_hash(
                    int.from_bytes(b"%c%c" % move, byteorder="big"), data["boards"][-1]
                )
            )

        for k in b"WRBDSM":
            if b"%c" % k in data["headers"]:
                del data["headers"][b"%c" % k]
        data["headers"] = {
            k.decode(): cls.unescape(v.decode()) for k, v in data["headers"].items()
        }

        return data

    @classmethod
    def encode(
        cls,
        boards: Iterable["BoardInfo"] = [],
        white_name: str = "?",
        black_name: str = "?",
        date: str = "?",
        result: GameState = GameState.NORMAL,
        headers: dict[str, str] = {},
    ):
        res = {
            b"M": b"",
            b"W": white_name.encode(),
            b"B": black_name.encode(),
            b"R": cls.RESULT_CODES[result],
        }
        res[b"D"] = date.encode()
        res.update({k.encode(): cls.escape(v).encode() for k, v in headers.items()})

        for move in get_moves(boards):
            res[b"M"] += hash(move).to_bytes(2, byteorder="big")

        return b"\n".join([k + b"\t" + v for k, v in res.items()])
