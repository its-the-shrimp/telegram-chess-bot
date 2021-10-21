import gzip
import itertools
import json
from telegram import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    InputMediaPhoto,
    InputMediaVideo,
    User,
    Message,
    Chat,
    Bot,
    Update,
)
import logging
import datetime
import threading
from . import core, media, analysis, utils, parsers, base
from typing import Any, List, Union, Optional

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


def get_pgn_file(update: Update, context, args: List) -> None:
    content = context.db.get(f"{args[1]}:pgn")
    update.effective_message.edit_reply_markup()
    if content:
        update.effective_message.reply_document(
            gzip.decompress(content),
            caption=context.langtable["pgn-file-caption"],
            filename=f"chess4u-{args[1]}.pgn",
        )
        context.db.delete(f"{args[1]}:pgn")
    else:
        update.callback_query.answer(
            context.langtable["pgn-fetch-error"], show_alert=True
        )


def from_bytes(src: bytes, bot: Bot, id: str) -> "BaseMatch":
    obj = parsers.CGNParser.decode(src)
    if "INLINE" in obj["headers"]:
        return GroupMatch._from_cgn(obj, bot, id)
    elif "difficulty" in obj["headers"]["OPTIONS"]:
        return AIMatch._from_cgn(obj, bot, id)
    else:
        return PMMatch._from_cgn(obj, bot, id)


class BaseMatch:
    ENGINE_FILENAME = None
    db = None

    def __init__(
        self,
        bot: Bot = None,
        id: str = None,
        options: dict[str, str] = {},
        date: str = None,
    ):
        self.init_msg_text: Optional[str] = None
        self.is_chess960: bool = options["ruleset"] == "chess960"
        self.options = options
        self.bot: Bot = bot
        self.states: list[core.BoardInfo] = [core.BoardInfo.from_fen(utils.STARTPOS)]
        self.result = "*"
        self.date = date or datetime.datetime.now().strftime(utils.DATE_FORMAT)
        self.id: str = id if id else base.create_match_id()
        self.video_filename: str = f"chess4u-{self.id}.mp4"
        self.image_filename: str = f"chess4u-{self.id}.jpg"

    def _keyboard(
        self,
        seq: list[dict[str, Union[str, int]]],
        expected_uid: int,
        handler_id: str = None,
        head_item: bool = False,
    ) -> Optional[InlineKeyboardMarkup]:
        res = []
        for button in seq:
            res.append(
                InlineKeyboardButton(
                    text=button["text"],
                    callback_data=base.format_callback_data(
                        button["data"][0],
                        args=button["data"][1:],
                        expected_uid=expected_uid,
                        handler_id=handler_id if handler_id else self.id,
                    ),
                )
            )

        if res:
            return InlineKeyboardMarkup(_group_items(res, 2, head_item=head_item))
        else:
            return None

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
                if self.states.count(test_board):
                    return "3fold-repetition-draw"
        if next(itertools.chain(*cur_side_moves), None) is None:
            return "stalemate-draw"

        pawns, knights, bishops, rooks, queens = [
            self.states[-1][cls] for cls in "pnbrq"
        ]
        if not rooks and not queens and not pawns:
            if any(
                [
                    len(bishops) == 1 and not knights,
                    len(knights) == 1 and not bishops,
                    len(bishops) == 2
                    and bishops[0].pos.is_lightsquare()
                    == bishops[1].pos.is_lightsquare()
                    and not knights,
                ]
            ):
                return "insufficient-material-draw"

        return "normal"

    def _send_analysis_video(self: Union["GroupMatch", "PMMatch"], lang_code: str, text: str):
        analyser = analysis.ChessEngine(BaseMatch.ENGINE_FILENAME)
        video, thumb = media.board_video(self, lang_code, analyser=analyser)
        new_msg = InputMediaVideo(
            base.get_tempfile_url(video, "video/mp4"), caption=text, thumb=thumb
        )
        self.db.set(
            f"{self.id}:pgn",
            gzip.compress(
                parsers.PGNParser.encode(
                    self.states,
                    white_name=self.db.get_name(self.player1),
                    black_name=self.db.get_name(self.player2),
                    date=self.date,
                    result=self.result,
                )
            ),
            ex=3600 * 48,
        )
        if issubclass(self, GroupMatch):
            self.msg = self.msg.edit_media(
                media=new_msg,
                reply_markup=self._keyboard(
                    [
                        {
                            "text": base.langtable[lang_code]["download-pgn"],
                            "data": ["DOWNLOAD", self.id],
                        }
                    ],
                    expected_uid=self.players[0].id,
                    handler_id="MAIN",
                ),
            )
        elif issubclass(self, PMMatch):
            if self.player_msg:
                self.player_msg = self.player_msg.edit_media(
                    media=new_msg,
                    reply_markup=self._keyboard(
                        [
                            {
                                "text": base.langtable[lang_code]["download-pgn"],
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
                                "text": base.langtable[lang_code]["download-pgn"],
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
            title_text = base.langtable[lang_code][state].format(
                name=self.db.get_name(player)
            )
            text = "\n".join(
                [
                    title_text,
                    base.langtable[lang_code]["winner"].format(
                        name=self.db.get_name(opponent)
                    ),
                    base.langtable[lang_code]["n-moves"].format(
                        n=self.states[-1].turn - 1
                    ),
                ]
            )
        else:
            title_text = base.langtable[lang_code][state]
            text = "\n".join(
                [
                    title_text,
                    base.langtable[lang_code]["is-draw"],
                    base.langtable[lang_code]["n-moves"].format(
                        n=self.states[-1].turn - 1
                    ),
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
            base.get_tempfile_url(
                media.board_image(
                    lang_code,
                    self.states,
                    player1_name=self.db.get_name(self.player1),
                    player2_name=self.db.get_name(self.player2),
                ),
                "image/jpeg",
            ),
            caption="\n".join(
                [
                    title_text,
                    base.langtable[lang_code]["waiting-eval"],
                ]
            ),
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
    def __init__(
        self,
        player1: User,
        player2: User,
        match_chat: int,
        msg: Union[Message, base.InlineMessageAdapter] = None,
        **kwargs,
    ):
        self.player1: User = player1
        self.player2: User = player2
        self.chat_id: int = match_chat
        self.msg: Union[Message, base.InlineMessageAdapter] = msg
        super().__init__(**kwargs)

    def __bytes__(self) -> bytes:
        return parsers.CGNParser.encode(
            self.states,
            white_name=self.db.get_name(self.player1),
            black_name=self.db.get_name(self.player2),
            date=self.date,
            result=self.result,
            headers={
                "WUID": hex(self.player1.id)[2:],
                "BUID": hex(self.player2.id)[2:],
                "CID": hex(self.chat_id)[2:],
                "MID": hex(self.msg.message_id)[2:],
                "MSGTXT": self.init_msg_text,
                "INLINE": str(int(isinstance(self.msg, base.InlineMessageAdapter))),
                "OPTIONS": json.dumps(self.options),
            },
        )

    @property
    def players(self) -> tuple[User, User]:
        return (
            (self.player1, self.player2)
            if self.states[-1].is_white_turn
            else (self.player2, self.player1)
        )

    @classmethod
    def _from_cgn(cls, obj: dict, bot: Bot, id: str) -> "GroupMatch":
        player1 = User(
            int(obj["headers"]["WUID"], 16), obj["white_name"], False, bot=bot
        )
        player2 = User(
            int(obj["headers"]["BUID"], 16), obj["black_name"], False, bot=bot
        )

        if bool(int(obj["headers"]["INLINE"])):
            msg = base.InlineMessageAdapter(obj["headers"]["MID"], bot)
        else:
            msg = Message(
                obj["headers"]["MID"],
                datetime.datetime.strptime(obj["date"], utils.DATE_FORMAT),
                Chat(int(obj["headers"]["CID"], 16, "group")),
                bot=bot,
            )

        return cls(
            player1,
            player2,
            int(obj["headers"]["CID"], 16),
            msg,
            options=json.loads(obj["headers"]["OPTIONS"]),
            bot=bot,
            id=id,
            date=obj["date"],
        )

    def init_turn(self, move: core.Move = None) -> None:
        super().init_turn(move=move)
        player, opponent = self.players
        state = self.get_state()
        langtable = base.langtable[player.language_code]
        if "draw" in state:
            self.result = "1/2-1/2"
        elif state == "checkmate":
            self.result = "0-1" if self.states[-1].is_white_turn else "1-0"

        if self.result != "*":
            self.send_analysis_video(player.language_code, state)
        else:
            msg = langtable["curr-move"].format(n=self.states[-1].turn)

            if state == "check":
                msg += langtable[player.language_code]["opponent-in-check"].format(
                    name=self.db.get_name(player)
                )
            else:
                msg += "\n"

            msg += langtable["opponent-to-move"].format(name=self.db.get_name(player))
            msg += "; " + langtable["player-to-move"]

            keyboard = self._keyboard(
                [
                    {"text": langtable["move-button"], "data": ["TURN"]},
                    {
                        "text": langtable["resign-button"],
                        "data": ["RESIGN"],
                    },
                ],
                player.id,
            )
            self.init_msg_text = msg
            img = media.board_image(
                player.language_code,
                self.states,
                player1_name=self.db.get_name(self.player1),
                player2_name=self.db.get_name(self.player2),
            )
            if self.msg:
                self.msg = self.msg.edit_media(
                    media=InputMediaPhoto(
                        base.get_tempfile_url(img, "image/jpeg"),
                        caption=msg,
                    ),
                    reply_markup=keyboard,
                )
            else:
                self.msg = self.bot.send_photo(
                    self.chat_id,
                    base.get_tempfile_url(img, "image/jpeg"),
                    caption=msg,
                    filename=self.image_filename,
                    reply_markup=keyboard,
                )

    def handle_input(self, command: str, args: list[Union[str, int, None]]) -> None:
        player, opponent = self.players
        langtable = base.langtable[player.language_code]
        allies, _ = self.pieces

        if command == "INIT_MSG":
            self.msg = self.msg.edit_caption(
                self.init_msg_text,
                reply_markup=self._keyboard(
                    [
                        {"text": langtable["move-button"], "data": ["TURN"]},
                        {
                            "text": langtable["resign-button"],
                            "data": ["RESIGN"],
                        },
                    ],
                    player.id,
                ),
            )

        if command == "TURN":
            piece_buttons = [{"text": langtable["back-button"], "data": ["INIT_MSG"]}]
            for piece in allies:
                if piece.get_moves():
                    piece_buttons.append(
                        {
                            "text": langtable["piece-desc"].format(
                                piece=langtable[type(piece).__name__.lower()],
                                pos=str(piece.pos),
                            ),
                            "data": ["CHOOSE_PIECE", str(piece.pos)],
                        }
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = langtable["opponent-to-move"].format(
                name=self.db.get_name(player)
            )
            new_text[-1] += "; " + langtable["player-to-choose-piece"]

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    base.get_tempfile_url(
                        media.board_image(
                            player.language_code,
                            self.states,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        "image/jpeg",
                    ),
                    caption="\n".join(new_text),
                ),
                reply_markup=self._keyboard(piece_buttons, player.id, head_item=True),
            )

        elif command == "RESIGN":
            self.result = "0-1" if self.states[-1].is_white_turn else "1-0"
            self.send_analysis_video(opponent.language_code, "resignation")

        elif command == "CHOOSE_PIECE":
            pos = utils.BoardPoint(args[0])
            dest_buttons = [{"text": langtable["back-button"], "data": ["TURN"]}]
            moves = self.states[-1][pos].get_moves()
            for move in moves:
                if move.is_promotion():
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + str(move.dst),
                            "data": ["PROMOTION_MENU", str(pos), str(move.dst)],
                        }
                    )
                else:
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + str(move.dst),
                            "data": ["MOVE", move.pgn_encode()],
                        }
                    )
            new_text = self.init_msg_text.split("\n")
            new_text[-1] = "; ".join(
                [
                    langtable["opponent-to-move"].format(name=self.db.get_name(player)),
                    langtable["player-to-choose-dest"],
                ]
            )

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    base.get_tempfile_url(
                        media.board_image(
                            player.language_code,
                            self.states,
                            selected=pos,
                            possible_moves=moves,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        "image/jpeg",
                    ),
                    caption="\n".join(new_text),
                ),
                reply_markup=self._keyboard(dest_buttons, player.id, head_item=True),
            )

        elif command == "PROMOTION_MENU":
            src = utils.BoardPoint(args[0])
            dst = utils.BoardPoint(args[1])
            move = core.Move.from_piece(self.states[-1][src], dst, new_piece="q")
            pieces = [
                {
                    "text": langtable["queen"],
                    "data": ["MOVE", move.pgn_encode()],
                },
                {
                    "text": langtable["knight"],
                    "data": ["MOVE", move.copy(new_piece="n").pgn_encode()],
                },
                {
                    "text": langtable["bishop"],
                    "data": ["MOVE", move.copy(new_piece="b").pgn_encode()],
                },
                {
                    "text": langtable["rook"],
                    "data": ["MOVE", move.copy(new_piece="r").pgn_encode()],
                },
            ]
            new_text = self.init_msg_text.split("\n")
            new_text[-1] = "; ".join(
                [
                    langtable["opponent-to-move"].format(name=self.db.get_name(player)),
                    langtable["player-to-promote"],
                ]
            )

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    base.get_tempfile_url(
                        media.board_image(
                            player.language_code,
                            self.states,
                            selected=src,
                            possible_moves=[move],
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        "image/jpeg",
                    ),
                    caption="\n".join(new_text),
                ),
                reply_markup=self._keyboard(pieces, player.id),
            )

        elif command == "MOVE":
            self.init_turn(move=core.Move.from_pgn(args[0], self.states[-1]))


class PMMatch(BaseMatch):
    def __init__(
        self,
        player1: User,
        player2: User,
        chat1: int,
        chat2: int,
        msg1: Message = None,
        msg2: Message = None,
        **kwargs,
    ):
        self.player1: User = player1
        self.player2: User = player2
        self.chat_id1: int = chat1
        self.chat_id2: int = chat2
        self.msg1: Message = msg1
        self.msg2: Message = msg2
        super().__init__(**kwargs)

    def __bytes__(self) -> bytes:
        return parsers.CGNParser.encode(
            self.states,
            white_name=self.db.get_name(self.player1),
            black_name=self.db.get_name(self.player2),
            date=self.date,
            result=self.result,
            headers={
                "WUID": hex(self.player1.id)[2:],
                "BUID": hex(self.player2.id)[2:],
                "CID1": hex(self.chat_id1)[2:],
                "CID2": hex(self.chat_id2)[2:],
                "MID1": hex(self.msg1.message_id)[2:],
                "MID2": hex(self.msg2.message_id)[2:],
                "MSGTXT": self.init_msg_text,
                "OPTIONS": json.dumps(self.options),
            },
        )

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

    @classmethod
    def _from_cgn(cls, obj: dict, bot: Bot, id: str) -> "PMMatch":
        date = datetime.datetime.strptime(obj["date"], utils.DATE_FORMAT)

        return cls(
            User(int(obj["headers"]["WUID"], 16), obj["white_name"], False, bot=bot),
            User(int(obj["headers"]["BUID"], 16), obj["black_name"], False, bot=bot),
            int(obj["headers"]["CID1"], 16),
            int(obj["headers"]["CID2"], 16),
            msg1=Message(
                int(obj["headers"]["MID1"], 16),
                date,
                Chat(int(obj["headers"]["CID1"], 16), "group"),
                text=obj["headers"]["MSGTXT"],
                bot=bot,
            ),
            msg2=Message(
                int(obj["headers"]["MID2"], 16),
                date,
                Chat(int(obj["headers"]["CID2"], 16), "group"),
                bot=bot,
            ),
            options=json.loads(obj["headers"]["OPTIONS"]),
            bot=bot,
            id=id,
            date=obj["date"],
        )

    def init_turn(self, move: core.Move = None, call_parent_method: bool = True):
        if call_parent_method:
            super().init_turn(move=move)
        player, opponent = self.players
        player_chatid, opponent_chatid = self.chat_ids
        player_langtable = base.langtable[player.language_code]
        opponent_langtable = base.langtable[opponent.language_code]
        state = self.get_state()
        if "draw" in state:
            self.result = "1/2-1/2"
        elif state == "checkmate":
            self.result = "0-1" if self.states[-1].is_white_turn else "1-0"

        if self.result != "*":
            self.send_analysis_video(player.language_code, state)
        else:
            player_msg = player_langtable["curr-move"].format(n=self.states[-1].turn)
            opponent_msg = opponent_langtable["curr-move"].format(
                n=self.states[-1].turn
            )

            if state == "check":
                player_msg += player_langtable["player-in-check"].format()
                opponent_msg += opponent_langtable["player-in-check"].format(
                    name=self.db.get_name(player)
                )
            else:
                player_msg = opponent_msg = player_msg + "\n"

            opponent_msg += opponent_langtable["opponent-to-move"].format(
                name=self.db.get_name(player)
            )
            player_msg += player_langtable["player-to-move"]

            self.init_msg_text = player_msg
            keyboard = self._keyboard(
                [
                    {"text": player_langtable["move-button"], "data": ["TURN"]},
                    {
                        "text": player_langtable["resign-button"],
                        "data": ["RESIGN"],
                    },
                ],
                player.id,
            )
            if self.player_msg:
                self.player_msg = self.player_msg.edit_media(
                    media=InputMediaPhoto(
                        base.get_tempfile_url(
                            media.board_image(
                                player.language_code,
                                self.states,
                                player1_name=self.db.get_name(self.player1),
                                player2_name=self.db.get_name(self.player2),
                            ),
                            "image/jpeg",
                        ),
                        caption=player_msg,
                    ),
                    reply_markup=keyboard,
                )
            elif player_chatid:
                self.player_msg = (
                    self.bot.send_photo(
                        player_chatid,
                        base.get_tempfile_url(
                            media.board_image(
                                player.language_code,
                                self.states,
                                player1_name=self.db.get_name(self.player1),
                                player2_name=self.db.get_name(self.player2),
                            ),
                            "image/jpeg",
                        ),
                        caption=player_msg,
                        reply_markup=keyboard,
                    )
                    if player_chatid
                    else None
                )

            if self.opponent_msg:
                self.opponent_msg = self.opponent_msg.edit_media(
                    media=InputMediaPhoto(
                        base.get_tempfile_url(
                            media.board_image(
                                opponent.language_code,
                                self.states,
                                player1_name=self.db.get_name(self.player1),
                                player2_name=self.db.get_name(self.player2),
                            ),
                            "image/jpeg",
                        ),
                        caption=opponent_msg,
                    )
                )
            elif opponent_chatid:
                self.opponent_msg = self.bot.send_photo(
                    opponent_chatid,
                    base.get_tempfile_url(
                        media.board_image(
                            opponent.language_code,
                            self.states,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        "image/jpeg",
                    ),
                    caption=opponent_msg,
                )

    def handle_input(self, command: str, args: list[Union[str, int, None]]):
        player, opponent = self.players
        langtable = base.langtable[player.language_code]
        allies, _ = self.pieces
        if command == "INIT_MSG":
            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    base.get_tempfile_url(
                        media.board_image(
                            player.language_code,
                            self.states,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        "image/jpeg",
                    ),
                    caption=self.init_msg_text,
                ),
                reply_markup=self._keyboard(
                    [
                        {"text": langtable["move-button"], "data": ["TURN"]},
                        {
                            "text": langtable["resign-button"],
                            "data": ["RESIGN"],
                        },
                    ],
                    player.id,
                ),
            )

        if command == "TURN":
            piece_buttons = [{"text": langtable["back-button"], "data": ["INIT_MSG"]}]
            for piece in allies:
                if piece.get_moves():
                    piece_buttons.append(
                        {
                            "text": langtable["piece-desc"].format(
                                piece=langtable[type(piece).__name__.lower()],
                                pos=str(piece.pos),
                            ),
                            "data": ["CHOOSE_PIECE", str(piece.pos)],
                        }
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = langtable["player-to-choose-piece"]

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    base.get_tempfile_url(
                        media.board_image(
                            player.language_code,
                            self.states,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        "image/jpeg",
                    ),
                    caption="\n".join(new_text),
                ),
                reply_markup=self._keyboard(piece_buttons, player.id, head_item=True),
            )

        elif command == "RESIGN":
            self.result = "0-1" if self.states[-1].is_white_turn else "1-0"
            self.send_analysis_video(player.language_code, "resignation")

        elif command == "CHOOSE_PIECE":
            pos = utils.BoardPoint(args[0])
            dest_buttons = [{"text": langtable["back-button"], "data": ["TURN"]}]
            moves = self.states[-1][pos].get_moves()
            for move in moves:
                if move.is_promotion():
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + str(move.dst),
                            "data": ["PROMOTION_MENU", str(pos), move.dst],
                        }
                    )
                else:
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + str(move.dst),
                            "data": ["MOVE", move.pgn_encode()],
                        }
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = langtable["player-to-choose-dest"]

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    base.get_tempfile_url(
                        media.board_image(
                            player.language_code,
                            self.states,
                            selected=pos,
                            possible_moves=moves,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        "image/jpeg",
                    ),
                    caption="\n".join(new_text),
                ),
                reply_markup=self._keyboard(dest_buttons, player.id, head_item=True),
            )

        elif command == "PROMOTION_MENU":
            src = utils.BoardPoint(args[0])
            dst = utils.BoardPoint(args[1])
            move = core.Move.from_piece(self.states[-1][src], dst, new_piece="q")
            pieces = [
                {
                    "text": langtable["queen"],
                    "data": ["MOVE", move.pgn_encode()],
                },
                {
                    "text": langtable["knight"],
                    "data": ["MOVE", move.copy(new_piece="n").pgn_encode()],
                },
                {
                    "text": langtable["bishop"],
                    "data": ["MOVE", move.copy(new_piece="b").pgn_encode()],
                },
                {
                    "text": langtable["rook"],
                    "data": ["MOVE", move.copy(new_piece="r").pgn_encode()],
                },
            ]

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = langtable["player-to-promote"]

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    base.get_tempfile_url(
                        media.board_image(
                            player.language_code,
                            self.states,
                            selected=src,
                            possible_moves=[move],
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        "image/jpeg",
                    ),
                    caption="\n".join(new_text),
                ),
                reply_markup=self._keyboard(pieces, player.id),
            )

        elif command == "MOVE":
            return self.init_turn(move=core.Move.from_pgn(args[0], self.states[-1]))


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

    def __bytes__(self) -> bytes:
        return parsers.CGNParser.encode(
            self.states,
            white_name=self.db.get_name(self.player1),
            date=self.date,
            result=self.result,
            headers={
                "WUID": hex(self.player1.id)[2:],
                "CID": hex(self.chat_id1)[2:],
                "MID": hex(self.msg1.message_id)[2:],
                "MSGTXT": self.init_msg_text,
                "OPTIONS": json.dumps(self.options),
            },
        )

    @classmethod
    def _from_cgn(cls, obj: dict, bot: Bot, id: str) -> "AIMatch":
        date = datetime.datetime.strptime(obj["date"], utils.DATE_FORMAT)

        return cls(
            User(int(obj["headers"]["WUID"], 16), obj["white_name"], False, bot=bot),
            int(obj["headers"]["CID"], 16),
            msg1=Message(
                int(obj["headers"]["MID"], 16),
                date,
                Chat(int(obj["headers"]["CID"], 16), "group"),
                text=obj["headers"]["MSGTXT"],
                bot=bot,
            ),
            options=json.loads(obj["headers"]["OPTIONS"]),
            bot=bot,
            id=id,
            date=obj["date"],
        )

    def init_turn(self, move: core.Move = None, **kwargs) -> None:
        super().init_turn(move=move, **kwargs)
        if move:
            return super().init_turn(
                self.engine.get_move(self.states[-1], self.EVAL_DEPTH)
            )
