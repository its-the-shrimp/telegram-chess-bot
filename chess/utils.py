from typing import Iterator, overload


ENGINE_FILENAME = "./stockfish_14_x64"
STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
DATE_FORMAT = "%Y.%m.%d"


def _reversed(src: dict):
    return {v: k for k, v in src.items()}


class BoardPoint:
    row: int
    column: int

    @overload
    def __new__(cls, src: str):
        ...

    @overload
    def __new__(cls, src: int):
        ...

    @overload
    def __new__(cls, column: int, row: int):
        ...

    def __new__(cls, *args):
        self = super().__new__(cls)
        if len(args) == 2:
            self.file, self.rank = args
        if len(args) == 1:
            if isinstance(args[0], str):
                if args[0] == "-":
                    return None
                self.file, self.rank = ord(args[0][0]) - 97, int(args[0][1]) - 1
            elif isinstance(args[0], int):
                self.rank, self.file = args[0] % 8, args[0] // 8

        return self

    def __bool__(self):
        return 0 <= self.rank <= 7 and 0 <= self.file <= 7

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.rank == other.rank
            and self.file == other.file
        )

    def __str__(self):
        return chr(self.file + 97) + str(self.rank + 1)

    def __repr__(self):
        return f"BoardPoint({self.__str__()})"

    def __index__(self):
        return self.file * 8 + self.rank

    def __iter__(self) -> Iterator:
        return (self.file, self.rank).__iter__()

    def copy(self, file: int = None, rank: int = None):
        return self.__class__(file or self.file, rank or self.rank)

    def is_lightsquare(self) -> bool:
        return self.file % 2 != self.rank % 2
