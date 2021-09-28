import gzip
import itertools
import random
from telegram import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    InputMediaPhoto,
    InputMediaVideo,
    User,
    Message,
    Chat,
    Bot,
)
import logging
import datetime
import threading
from .utils import *
from . import core, media, analysis
from typing import Any, List, Union, Optional

IDSAMPLE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-+"
MOVETYPE_MARKERS = {"normal": "", "killing": "❌", "castling": "🔀", "promotion": "⏫"}
JSON = dict[str, Union[str, dict]]


def _group_items(obj: list[Any], n: int, head_item: bool = False) -> list[list[Any]]:
    res = []
    for index in range(len(obj)):
        index -= int(head_item)
        if index == -1:
            res.append([obj[0]])
        elif index // n == index / n:
            res.append([obj[index + int(head_item)]])
        else:
            res[-1].append(obj[index + int(head_item)])

    return res


def from_dict(obj: dict[str, Any], match_id: str, bot: Bot) -> "BaseMatch":
    cls = eval(obj["type"] + "Match")
    return cls.from_dict(obj, match_id, bot)


class BaseMatch:
    ENGINE_FILENAME = "./stockfish_14_x64"
    db = None

    def __init__(self, bot=None, id=None, options: dict[str, str] = {}):
        self.init_msg_text: Optional[str] = None
        self.is_chess960: bool = options["ruleset"] == "chess960"
        self.options = options
        self.bot: Bot = bot
        self.states: list[core.BoardInfo] = [core.BoardInfo.from_fen(STARTPOS)]
        self.result = "*"
        self.id: str = (
            id if id else "".join([random.choice(IDSAMPLE) for i in range(8)])
        )
        self.image_filename: str = f"chess4u-{self.id}.jpg"
        self.video_filename: str = f"chess4u-{self.id}.mp4"

    def _keyboard(
        self,
        seq: list[dict[str, Union[str, BoardPoint]]],
        expected_uid: int,
        handler_id: str = None,
        head_item: bool = False,
    ) -> Optional[InlineKeyboardMarkup]:
        res = []
        for button in seq:
            data = []
            for argument in button["data"]:
                if type(argument) == BoardPoint:
                    data.append(encode_pos(argument))
                elif argument == None:
                    data.append("")
                else:
                    data.append(str(argument))
            res.append(
                InlineKeyboardButton(
                    text=button["text"],
                    callback_data=f"{expected_uid if expected_uid else ''}\n{handler_id or self.id}\n{'#'.join(data)}",
                )
            )

        if res:
            return InlineKeyboardMarkup(_group_items(res, 2, head_item=head_item))
        else:
            return None

    def to_dict(self) -> JSON:
        return {
            "type": "Base",
            "moves": core.get_pgn_moveseq(core.get_moves(self.states)),
            "options": self.options,
        }

    @property
    def pieces(self) -> tuple[tuple[core.BasePiece]]:
        return (
            (self.states[-1].whites, self.states[-1].blacks)
            if self.states[-1].is_white_turn
            else (self.states[-1].blacks, self.states[-1].whites)
        )

    def get_state(self) -> Optional[str]:
        cur_king = self.states[-1]["k", self.states[-1].is_white_turn][0]
        if cur_king.in_checkmate():
            return "checkmate"
        if cur_king.in_check():
            return "check"

        if self.states[-1].empty_halfturns >= 50:
            return "50-move-draw"
        cur_side_moves = [piece.get_moves() for piece in cur_king.allied_pieces]
        if len(self.states) >= 8:
            for move in itertools.chain(
                *(
                    [piece.get_moves() for piece in cur_king.enemy_pieces]
                    + cur_side_moves
                )
            ):
                test_board = self.states[-1] + move
                if test_board == self.states[-8]:
                    return "3fold-repetition-draw"
        if next(itertools.chain(*cur_side_moves), None) is None:
            return "stalemate-draw"

        pawns, knights, bishops, rooks, queens = [
            self.states[-1][cls] for cls in ["p", "n", "b", "r", "q"]
        ]
        if not rooks and not queens and not pawns:
            if any(
                [
                    len(bishops) == 1 and not knights,
                    len(knights) == 1 and not bishops,
                    len(bishops) == 2
                    and is_lightsquare(bishops[0].pos) == is_lightsquare(bishops[1].pos)
                    and not knights,
                ]
            ):
                return "insufficient-material-draw"

        return "normal"

    def pgn_encode(self, headers: dict = {}):
        std_headers = {
            "Event": "Online Chess on Telegram",
            "Site": "t.me/real_chessbot",
            "Date": datetime.datetime.now().strftime("%Y.%m.%d"),
            "Round": 1,
            "White": self.db.get_name(self.player1)
            if self.db and hasattr(self, "player1")
            else "?",
            "Black": self.db.get_name(self.player2)
            if self.db and hasattr(self, "player2")
            else "?",
            "Result": self.result,
        }
        headers = std_headers | headers

        encoded = "\n".join([f'[{k} "{v}"]' for k, v in headers.items()])
        return (
            encoded
            + "\n\n"
            + core.get_pgn_moveseq(core.get_moves(self.states), result=self.result)
        )

    def _send_analysis_video(self, lang_code: str, text: str):
        analyser = analysis.ChessEngine(BaseMatch.ENGINE_FILENAME)
        video, thumb = media.board_video(self, lang_code, analyser=analyser)
        new_msg = InputMediaVideo(
            video, caption=text, filename=self.video_filename, thumb=thumb
        )
        self.db.set(
            f"{self.id}:pgn", gzip.compress(self.pgn_encode().encode()), ex=3600 * 48
        )
        if hasattr(self, "msg"):
            self.msg = self.msg.edit_media(
                media=new_msg,
                reply_markup=self._keyboard(
                    [
                        {
                            "text": langtable[lang_code]["download-pgn"],
                            "data": ["DOWNLOAD", self.id],
                        }
                    ],
                    expected_uid=self.players[0].id,
                    handler_id="MAIN",
                ),
            )
        elif hasattr(self, "player_msg"):
            if self.player_msg:
                self.player_msg = self.player_msg.edit_media(
                    media=new_msg,
                    reply_markup=self._keyboard(
                        [
                            {
                                "text": langtable[lang_code]["download-pgn"],
                                "data": ["DOWNLOAD", self.id],
                            }
                        ],
                        expected_uid=self.players[0].id,
                        handler_id="MAIN",
                    ),
                )
            if self.opponent_msg:
                self.opponent_msg = self.opponent_msg.edit_media(
                    media=new_msg,
                    reply_markup=self._keyboard(
                        [
                            {
                                "text": langtable[lang_code]["download-pgn"],
                                "data": ["DOWNLOAD", self.id],
                            }
                        ],
                        expected_uid=self.players[1].id,
                        handler_id="MAIN",
                    ),
                )
        else:
            raise ValueError

    def send_analysis_video(self, lang_code: str, state: str) -> None:
        if state in ("checkmate", "resignation"):
            player, opponent = self.players
            title_text = langtable[lang_code][state].format(
                name=self.db.get_name(player)
            )
            text = "\n".join(
                [
                    title_text,
                    langtable[lang_code]["winner"].format(
                        name=self.db.get_name(opponent)
                    ),
                    langtable[lang_code]["n-moves"].format(n=self.states[-1].turn - 1),
                ]
            )
        else:
            title_text = langtable[lang_code][state]
            text = "\n".join(
                [
                    title_text,
                    langtable[lang_code]["is-draw"],
                    langtable[lang_code]["n-moves"].format(n=self.states[-1].turn - 1),
                ]
            )

        eval_thread = threading.Thread(
            name=self.id + ":eval",
            target=self._send_analysis_video,
            args=(
                lang_code,
                text,
            ),
        )
        eval_thread.start()
        last_boardimg = InputMediaPhoto(
            media.board_image(
                lang_code,
                self.states,
                player1_name=self.db.get_name(self.player1),
                player2_name=self.db.get_name(self.player2),
            ),
            caption="\n".join(
                [
                    title_text,
                    langtable[lang_code]["waiting-eval"],
                ]
            ),
            filename=self.image_filename,
        )
        if hasattr(self, "msg"):
            self.msg = self.msg.edit_media(media=last_boardimg)
        elif hasattr(self, "player_msg"):
            if self.player_msg:
                self.player_msg = self.player_msg.edit_media(last_boardimg)
            if self.opponent_msg:
                self.opponent_msg = self.opponent_msg.edit_media(last_boardimg)

    def init_turn(self, move: core.Move = None) -> None:
        logging.debug("Move made: " + (move.pgn_encode() if move else ""))
        if move:
            self.states.append(self.states[-1] + move)


class GroupMatch(BaseMatch):
    def __init__(self, player1: User, player2: User, match_chat: int, **kwargs):
        self.player1: User = player1
        self.player2: User = player2
        self.chat_id: int = match_chat
        self.msg: Message = None
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, obj: JSON, match_id: int, bot: Bot) -> "GroupMatch":
        logging.debug(f"Constructing {cls.__name__} object:", obj)
        player1 = User.de_json(obj["player1"] | {"is_bot": False}, bot)
        player2 = User.de_json(obj["player2"] | {"is_bot": False}, bot)
        new = cls(
            player1,
            player2,
            obj["chat_id"],
            bot=bot,
            id=match_id,
        )
        new.states, new.result = core.decode_pgn_moveseq(obj["moves"])
        new.init_msg_text = obj["msg_text"]
        new.msg = Message(
            obj["msg_id"],
            datetime.datetime.now().timestamp(),
            Chat(obj["chat_id"], "group", bot=bot),
            bot=bot,
            caption=obj["msg_text"],
        )

        return new

    @property
    def players(self) -> tuple[User, User]:
        return (
            (self.player1, self.player2)
            if self.states[-1].is_white_turn
            else (self.player2, self.player1)
        )

    def to_dict(self) -> JSON:
        res = super().to_dict()
        res.update(
            {
                "type": "Group",
                "chat_id": self.chat_id,
                "msg_id": self.msg.message_id,
                "msg_text": self.init_msg_text,
                "player1": {
                    k: v
                    for k, v in self.player1.to_dict().items()
                    if k in ["username", "id", "first_name", "last_name"]
                },
                "player2": {
                    k: v
                    for k, v in self.player2.to_dict().items()
                    if k in ["username", "id", "first_name", "last_name"]
                },
            }
        )
        return res

    def init_turn(self, lang_code, move: core.Move = None) -> None:
        super().init_turn(move=move)
        player, opponent = self.players
        state = self.get_state()
        if "draw" in state:
            self.result = "1/2-1/2"
        elif state == "checkmate":
            self.result = "0-1" if self.states[-1].is_white_turn else "1-0"

        if self.result != "*":
            self.send_analysis_video(lang_code, state)
        else:
            msg = langtable[lang_code]["curr-move"].format(n=self.states[-1].turn)

            if state == "check":
                msg += langtable[lang_code]["opponent-in-check"].format(
                    name=self.db.get_name(player)
                )
            else:
                msg += "\n"

            msg += langtable[lang_code]["opponent-to-move"].format(
                name=self.db.get_name(player)
            )
            msg += "; " + langtable[lang_code]["player-to-move"]

            keyboard = self._keyboard(
                [
                    {"text": langtable[lang_code]["move-button"], "data": ["TURN"]},
                    {
                        "text": langtable[lang_code]["resign-button"],
                        "data": ["SURRENDER"],
                    },
                ],
                player.id,
            )
            self.init_msg_text = msg
            img = media.board_image(
                lang_code,
                self.states,
                player1_name=self.db.get_name(self.player1),
                player2_name=self.db.get_name(self.player2),
            )
            if self.msg:
                self.msg = self.msg.edit_media(
                    media=InputMediaPhoto(
                        img,
                        caption=msg,
                        filename=self.image_filename,
                    ),
                    reply_markup=keyboard,
                )
            else:
                self.msg = self.bot.send_photo(
                    self.chat_id,
                    img,
                    caption=msg,
                    filename=self.image_filename,
                    reply_markup=keyboard,
                )

    def handle_input(self, lang_code: str, args: list[Union[str, int]]) -> None:
        player, opponent = self.players
        allies, _ = self.pieces

        if args[0] == "INIT_MSG":
            self.msg = self.msg.edit_caption(
                self.init_msg_text,
                reply_markup=self._keyboard(
                    [
                        {"text": langtable[lang_code]["move-button"], "data": ["TURN"]},
                        {
                            "text": langtable[lang_code]["resign-button"],
                            "data": ["SURRENDER"],
                        },
                    ],
                    player.id,
                ),
            )

        if args[0] == "TURN":
            piece_buttons = [
                {"text": langtable[lang_code]["back-button"], "data": ["INIT_MSG"]}
            ]
            for piece in allies:
                if piece.get_moves():
                    piece_buttons.append(
                        {
                            "text": langtable[lang_code]["piece-desc"].format(
                                piece=langtable[lang_code][
                                    type(piece).__name__.lower()
                                ],
                                pos=encode_pos(piece.pos),
                            ),
                            "data": ["CHOOSE_PIECE", piece.pos],
                        }
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = langtable[lang_code]["opponent-to-move"].format(
                name=self.db.get_name(player)
            )
            new_text[-1] += "; " + langtable[lang_code]["player-to-choose-piece"]

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        lang_code,
                        self.states,
                        player1_name=self.db.get_name(self.player1),
                        player2_name=self.db.get_name(self.player2),
                    ),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(piece_buttons, player.id, head_item=True),
            )

        elif args[0] == "SURRENDER":
            self.result = "0-1" if self.states[-1].is_white_turn else "1-0"
            self.send_analysis_video(lang_code, "resignation")

        elif args[0] == "CHOOSE_PIECE":
            args[1] = decode_pos(args[1])
            dest_buttons = [
                {"text": langtable[lang_code]["back-button"], "data": ["TURN"]}
            ]
            moves = self.states[-1][args[1]].get_moves()
            for move in moves:
                if move.is_promotion():
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + encode_pos(move.dst),
                            "data": ["PROMOTION_MENU", args[1], move.dst],
                        }
                    )
                else:
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + encode_pos(move.dst),
                            "data": ["MOVE", move.pgn_encode()],
                        }
                    )
            new_text = self.init_msg_text.split("\n")
            new_text[-1] = "; ".join(
                [
                    langtable[lang_code]["opponent-to-move"].format(
                        name=self.db.get_name(player)
                    ),
                    langtable[lang_code]["player-to-choose-dest"],
                ]
            )

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        lang_code,
                        self.states,
                        selected=args[1],
                        possible_moves=moves,
                        player1_name=self.db.get_name(self.player1),
                        player2_name=self.db.get_name(self.player2),
                    ),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(dest_buttons, player.id, head_item=True),
            )

        elif args[0] == "PROMOTION_MENU":
            move = core.Move.from_piece(
                self.states[-1][args[1]], args[2], new_piece="q"
            )
            pieces = [
                {
                    "text": langtable[lang_code]["queen"],
                    "data": ["MOVE", move.pgn_encode()],
                },
                {
                    "text": langtable[lang_code]["knight"],
                    "data": ["MOVE", move.copy(new_piece="n").pgn_encode()],
                },
                {
                    "text": langtable[lang_code]["bishop"],
                    "data": ["MOVE", move.copy(new_piece="b").pgn_encode()],
                },
                {
                    "text": langtable[lang_code]["rook"],
                    "data": ["MOVE", move.copy(new_piece="r").pgn_encode()],
                },
            ]
            new_text = self.init_msg_text.split("\n")
            new_text[-1] = "; ".join(
                [
                    langtable[lang_code]["opponent-to-move"].format(
                        name=self.db.get_name(player)
                    ),
                    langtable[lang_code]["player-to-promote"],
                ]
            )

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        lang_code,
                        self.states,
                        selected=args[1],
                        possible_moves=[move],
                        player1_name=self.db.get_name(self.player1),
                        player2_name=self.db.get_name(self.player2),
                    ),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(pieces, player.id),
            )

        elif args[0] == "MOVE":
            self.init_turn(lang_code, move=core.Move.from_pgn(args[1], self.states[-1]))


class PMMatch(BaseMatch):
    def __init__(self, player1: User, player2: User, chat1: int, chat2: int, **kwargs):
        self.player1: User = player1
        self.player2: User = player2
        self.chat_id1: int = chat1
        self.chat_id2: int = chat2
        self.msg1: Message = None
        self.msg2: Message = None
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, obj: JSON, match_id: int, bot=Bot) -> "PMMatch":
        logging.debug(f"Constructing {cls.__name__} object:", obj)
        player1 = User.de_json(obj["player1"] | {"is_bot": False}, bot)
        player2 = User.de_json(obj["player2"] | {"is_bot": False}, bot)
        new = cls(
            player1,
            player2,
            obj["chat_id1"],
            obj["chat_id2"],
            bot=bot,
            id=match_id,
        )
        new.states, new.result = core.decode_pgn_moveseq(obj["moves"])
        new.init_msg_text = obj["msg_text"]
        new.msg1 = Message(
            obj["msg_id1"],
            datetime.datetime.now().timestamp(),
            Chat(obj["chat_id1"], "private", bot=bot),
            bot=bot,
            caption=obj["msg_text"],
        )
        new.msg2 = Message(
            obj["msg_id2"],
            datetime.datetime.now().timestamp(),
            Chat(obj["chat_id2"], "private", bot=bot),
            bot=bot,
        )

        return new

    @property
    def player_msg(self) -> Message:
        return self.msg1 if self.states[-1].is_white_turn else self.msg2

    @player_msg.setter
    def player_msg(self, msg: Message) -> None:
        if self.states[-1].is_white_turn:
            self.msg1 = msg
        else:
            self.msg2 = msg

    @property
    def opponent_msg(self) -> Message:
        return self.msg2 if self.states[-1].is_white_turn else self.msg1

    @opponent_msg.setter
    def opponent_msg(self, msg: Message) -> None:
        if self.states[-1].is_white_turn:
            self.msg2 = msg
        else:
            self.msg1 = msg

    @property
    def players(self) -> tuple[User, User]:
        return (
            (self.player1, self.player2)
            if self.states[-1].is_white_turn
            else (self.player2, self.player1)
        )

    @property
    def chat_ids(self) -> tuple[int, int]:
        return (
            (self.chat_id1, self.chat_id2)
            if self.states[-1].is_white_turn
            else (self.chat_id2, self.chat_id1)
        )

    def to_dict(self) -> JSON:
        res = super().to_dict()
        res.update(
            {
                "type": "PM",
                "chat_id1": self.chat_id1,
                "chat_id2": self.chat_id2,
                "msg_id1": self.msg1.message_id,
                "msg_id2": getattr(self.msg2, "message_id", None),
                "msg_text": self.init_msg_text,
                "player1": {
                    k: v
                    for k, v in self.player1.to_dict().items()
                    if k in ["username", "id", "first_name", "last_name"]
                },
                "player2": {
                    k: v
                    for k, v in self.player2.to_dict().items()
                    if k in ["username", "id", "first_name", "last_name"]
                },
            }
        )
        return res

    def init_turn(
        self, lang_code: str, move: core.Move = None, call_parent_method: bool = True
    ):
        if call_parent_method:
            super().init_turn(move=move)
        player, opponent = self.players
        player_chatid, opponent_chatid = self.chat_ids
        state = self.get_state()
        if "draw" in state:
            self.result = "1/2-1/2"
        elif state == "checkmate":
            self.result = "0-1" if self.states[-1].is_white_turn else "1-0"

        if self.result != "*":
            self.send_analysis_video(lang_code, state)
        else:
            player_msg = opponent_msg = langtable[lang_code]["curr-move"].format(
                n=self.states[-1].turn
            )

            if state == "check":
                player_msg += langtable[lang_code]["player-in-check"].format()
                opponent_msg += langtable[lang_code]["player-in-check"].format(
                    name=self.db.get_name(player)
                )
            else:
                player_msg = opponent_msg = player_msg + "\n"

            opponent_msg += langtable[lang_code]["opponent-to-move"].format(
                name=self.db.get_name(player)
            )
            player_msg += langtable[lang_code]["player-to-move"]

            self.init_msg_text = player_msg
            keyboard = self._keyboard(
                [
                    {"text": langtable[lang_code]["move-button"], "data": ["TURN"]},
                    {
                        "text": langtable[lang_code]["resign-button"],
                        "data": ["SURRENDER"],
                    },
                ],
                player.id,
            )
            if self.player_msg:
                self.player_msg = self.player_msg.edit_media(
                    media=InputMediaPhoto(
                        media.board_image(
                            lang_code,
                            self.states,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        caption=player_msg,
                        filename=self.image_filename,
                    ),
                    reply_markup=keyboard,
                )
            elif player_chatid:
                self.player_msg = (
                    self.bot.send_photo(
                        player_chatid,
                        media.board_image(
                            lang_code,
                            self.states,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        caption=player_msg,
                        filename=self.image_filename,
                        reply_markup=keyboard,
                    )
                    if player_chatid
                    else None
                )

            if self.opponent_msg:
                self.opponent_msg = self.opponent_msg.edit_media(
                    media=InputMediaPhoto(
                        media.board_image(
                            lang_code,
                            self.states,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        caption=opponent_msg,
                        filename=self.image_filename,
                    )
                )
            elif opponent_chatid:
                self.opponent_msg = self.bot.send_photo(
                    opponent_chatid,
                    media.board_image(
                        lang_code,
                        self.states,
                        player1_name=self.db.get_name(self.player1),
                        player2_name=self.db.get_name(self.player2),
                    ),
                    caption=opponent_msg,
                    filename=self.image_filename,
                )

    def handle_input(self, lang_code: str, args: List):
        player, opponent = self.players
        allies, _ = self.pieces
        if args[0] == "INIT_MSG":
            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        lang_code,
                        self.states,
                        player1_name=self.db.get_name(self.player1),
                        player2_name=self.db.get_name(self.player2),
                    ),
                    caption=self.init_msg_text,
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(
                    [
                        {"text": langtable[lang_code]["move-button"], "data": ["TURN"]},
                        {
                            "text": langtable[lang_code]["resign-button"],
                            "data": ["SURRENDER"],
                        },
                    ],
                    player.id,
                ),
            )

        if args[0] == "TURN":
            piece_buttons = [
                {"text": langtable[lang_code]["back-button"], "data": ["INIT_MSG"]}
            ]
            for piece in allies:
                if piece.get_moves():
                    piece_buttons.append(
                        {
                            "text": langtable[lang_code]["piece-desc"].format(
                                piece=langtable[lang_code][
                                    type(piece).__name__.lower()
                                ],
                                pos=encode_pos(piece.pos),
                            ),
                            "data": ["CHOOSE_PIECE", piece.pos],
                        }
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = langtable[lang_code]["player-to-choose-piece"]

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        lang_code,
                        self.states,
                        player1_name=self.db.get_name(self.player1),
                        player2_name=self.db.get_name(self.player2),
                    ),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(piece_buttons, player.id, head_item=True),
            )

        elif args[0] == "SURRENDER":
            self.result = "0-1" if self.states[-1].is_white_turn else "1-0"
            self.send_analysis_video(lang_code, "resignation")

        elif args[0] == "CHOOSE_PIECE":
            args[1] = decode_pos(args[1])
            dest_buttons = [
                {"text": langtable[lang_code]["back-button"], "data": ["TURN"]}
            ]
            moves = self.states[-1][args[1]].get_moves()
            for move in moves:
                if move.is_promotion():
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + encode_pos(move.dst),
                            "data": ["PROMOTION_MENU", args[1], move.dst],
                        }
                    )
                else:
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + encode_pos(move.dst),
                            "data": ["MOVE", move.pgn_encode()],
                        }
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = langtable[lang_code]["player-to-choose-dest"]

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        lang_code,
                        self.states,
                        selected=args[1],
                        possible_moves=moves,
                        player1_name=self.db.get_name(self.player1),
                        player2_name=self.db.get_name(self.player2),
                    ),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(dest_buttons, player.id, head_item=True),
            )

        elif args[0] == "PROMOTION_MENU":
            args[1] = decode_pos(args[1])
            args[2] = decode_pos(args[2])
            move = core.Move.from_piece(
                self.states[-1][args[1]], args[2], new_piece="q"
            )
            pieces = [
                {
                    "text": langtable[lang_code]["queen"],
                    "data": ["MOVE", move.pgn_encode()],
                },
                {
                    "text": langtable[lang_code]["knight"],
                    "data": ["MOVE", move.copy(new_piece="n").pgn_encode()],
                },
                {
                    "text": langtable[lang_code]["bishop"],
                    "data": ["MOVE", move.copy(new_piece="b").pgn_encode()],
                },
                {
                    "text": langtable[lang_code]["rook"],
                    "data": ["MOVE", move.copy(new_piece="r").pgn_encode()],
                },
            ]

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = langtable[lang_code]["player-to-promote"]

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        lang_code,
                        self.states,
                        selected=args[1],
                        possible_moves=[move],
                        player1_name=self.db.get_name(self.player1),
                        player2_name=self.db.get_name(self.player2),
                    ),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(pieces, player.id),
            )

        elif args[0] == "MOVE":
            return self.init_turn(
                lang_code, move=core.Move.from_pgn(args[1], self.states[-1])
            )


class AIMatch(PMMatch):
    DIFF_PRESETS = {
        "low-diff": (0, 1, 1, 2, 2, 2, 2),
        "mid-diff": (0, 1, 1, 2, 2),
        "high-diff": (0, 1, 2),
        "max-diff": (0,),
    }
    EVAL_DEPTH = 18

    def __init__(
        self, player: User, chat_id: int, options: dict[str, str] = {}, **kwargs
    ):
        super().__init__(
            player, kwargs["bot"].get_me(), chat_id, 0, options=options, **kwargs
        )
        self.engine = analysis.ChessEngine(self.ENGINE_FILENAME)
        self.difficulty = options["difficulty"]

        self.engine.set_move_probabilities(self.DIFF_PRESETS[self.difficulty])
        self.engine["UCI_Chess960"] = self.is_chess960

    @classmethod
    def from_dict(cls, obj: JSON, match_id: int, bot=Bot) -> "AIMatch":
        logging.debug(f"Constructing {cls.__name__} object: {obj}")
        player = User.de_json(obj["player1"] | {"is_bot": False}, bot)
        new = cls(player, obj["chat_id1"], bot=bot, id=match_id, options=obj["options"])
        new.states, new.result = core.decode_pgn_moveseq(obj["moves"])
        new.init_msg_text = obj["msg_text"]
        new.engine.set_move_probabilities()
        new.msg1 = Message(
            obj["msg_id1"],
            datetime.datetime.now().timestamp(),
            Chat(obj["chat_id1"], "private", bot=bot),
            bot=bot,
            caption=obj["msg_text"],
        )

        return new

    def to_dict(self) -> JSON:
        res = super().to_dict()
        del res["player2"], res["msg_id2"], res["chat_id2"]
        res["type"] = "AI"
        return res

    def init_turn(self, lang_code: str, move: core.Move = None, **kwargs) -> None:
        super().init_turn(lang_code, move=move, **kwargs)
        if move:
            return super().init_turn(
                lang_code, self.engine.get_move(self.states[-1], self.EVAL_DEPTH)
            )
