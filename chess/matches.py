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
    CallbackQuery
)
import logging
import datetime
import threading

from telegram.ext import Dispatcher
from . import core, media, analysis, utils, parsers, base
from typing import Any, Iterator, TypedDict, Union, Optional, cast

MOVETYPE_MARKERS = {"normal": "", "killing": "❌", "castling": "🔀", "promotion": "⏫"}
JSON = dict[str, Union[str, dict]]
_CallbackDataInput = TypedDict("_CallbackDataInput", {"text": str, "data": list[str]})


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


def _get_pgn_file(update: Update, context: base.BoardGameContext, args: tuple):
    content, match_id = args
    cast(Chat, update.effective_chat).send_document(
        content,
        caption=context.langtable["pgn-file-caption"],
        filename=f"chess4u-{match_id}.pgn",
    )


def get_pgn_file(update: Update, context: base.BoardGameContext, args: list[str]) -> None:
    raw = context.db.get(f"{args[0]}:pgn")
    if update.effective_message is not None:
        update.effective_message.edit_reply_markup()

    if raw is not None:
        
        content = gzip.decompress(raw)
        if update.effective_message is not None:
            update.effective_message.reply_document(
                content,
                caption=context.langtable["pgn-file-caption"],
                filename=f"chess4u-{args[0]}.pgn",
            )
        else:
            msgurl = context.db.set_pending_message(
                _get_pgn_file,
                args=(content, args[0]),
            )
            cast(CallbackQuery, update.callback_query).answer(url=msgurl)
        context.db.delete(f"{args[0]}:pgn")
    else:
        cast(CallbackQuery, update.callback_query).answer(
            context.langtable["pgn-fetch-error"], show_alert=True
        )





def from_bytes(src: bytes, dispatcher: Dispatcher, id: str) -> "BaseMatch":
    obj = parsers.CGNParser.decode(src)
    if "INLINE" in obj["headers"]:
        return GroupMatch._from_cgn(obj, dispatcher, id)
    elif "difficulty" in obj["headers"]["OPTIONS"]:
        return AIMatch._from_cgn(obj, dispatcher, id)
    else:
        return PMMatch._from_cgn(obj, dispatcher, id)


class BaseMatch:
    ENGINE_FILENAME: str = "./stockfish_14_x64"
    db: base.RedisInterface
    player1: User
    player2: User

    def __init__(
        self,
        dispatcher: Dispatcher = None,
        id: str = None,
        options: dict[str, str] = {"ruleset": "chess960", "timectrl": "classic"},
        date: str = None,
    ):
        self.init_msg_text: str = ""
        self.is_chess960: bool = options["ruleset"] == "chess960"
        self.options: dict[str, str] = options
        self.dispatcher: Optional[Dispatcher] = dispatcher
        self.startpos: core.BoardInfo = (
            core.BoardInfo.init_chess960()
            if self.is_chess960
            else core.BoardInfo.from_fen(options.get("pos", utils.STARTPOS))
        )
        self.moves: list[core.Move] = []
        self.state: core.GameState = core.GameState.NORMAL
        self.date: Optional[str] = date or datetime.datetime.now().strftime(utils.DATE_FORMAT)
        self.id: str = id or base.create_match_id()
        self.draw_offered: bool = False

    def __contains__(self, user: User):
        return user == self.player1 or user == self.player2

    def __repr__(self):
        return f"<{self.__class__.__name__}({parsers.PGNParser.encode_moveseq(moves=self.moves, result=self.state)})>"

    @property
    def bot(self) -> Bot:
        return cast(Dispatcher, self.dispatcher).bot

    @property
    def cur_board(self):
        return self.moves[-1].apply() if self.moves else self.startpos

    def _keyboard(
        self,
        seq: list[_CallbackDataInput],
        expected_uid: int = None,
        handler_id: str = None,
        head_item: bool = False,
    ) -> Optional[InlineKeyboardMarkup]:
        res = []
        for button in seq:
            res.append(
                InlineKeyboardButton(
                    text=button["text"],
                    callback_data=str(base.CallbackData(
                        button["data"][0],
                        args=button["data"][1:],
                        expected_uid=expected_uid,
                        handler_id=handler_id or self.id,
                    )),
                )
            )

        if res:
            return InlineKeyboardMarkup(_group_items(res, 2, head_item=head_item))
        else:
            return None

    @property
    def pieces(self) -> tuple[Iterator[core.BasePiece], Iterator[core.BasePiece]]:
        return (
            (self.cur_board.whites, self.cur_board.blacks)
            if self.cur_board.is_white_turn
            else (self.cur_board.blacks, self.cur_board.whites)
        )

    @property
    def players(self) -> tuple[User, User]:
        return (
            (self.player1, self.player2)
            if self.cur_board.is_white_turn
            else (self.player2, self.player1)
        )

    def get_state(self) -> core.GameState:
        cur_king = self.cur_board.get_king(self.cur_board.is_white_turn)
        if cur_king.in_checkmate():
            return (
                core.GameState.WHITE_CHECKMATED
                if self.cur_board.is_white_turn
                else core.GameState.BLACK_CHECKMATED
            )
        if cur_king.in_check():
            return core.GameState.CHECK

        if self.cur_board.empty_halfturns >= 50:
            return core.GameState.FIFTY_MOVE_DRAW
        cur_side_moves = [piece.get_moves() for piece in cur_king.allied_pieces]
        if len(self.moves) >= 7:
            for move in itertools.chain(
                *(
                    [piece.get_moves() for piece in cur_king.enemy_pieces]
                    + cur_side_moves
                )
            ):
                if self.moves.count(move) == 3:
                    return core.GameState.THREEFOLD_REPETITION_DRAW
        if next(itertools.chain(*cur_side_moves), None) is None:
            return core.GameState.STALEMATE_DRAW

        pawns, knights, bishops, rooks, queens = [
            self.cur_board.get_by_type(cls, None)
            for cls in (core.Pawn, core.Knight, core.Bishop, core.Rook, core.Queen)
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
                return core.GameState.INSUFFICIENT_MATERIAL_DRAW

        return core.GameState.NORMAL

    def _send_analysis_video(self, lang_code: str, text: str):
        try:
            analyser = analysis.ChessEngine(
                BaseMatch.ENGINE_FILENAME, default_eval_depth=18
            )
            for index in range(len(self.moves)):
                analyser.eval_move(self.moves[index])
            video, thumb = media.board_video(self, lang_code)
            logging.info("Analysis video created")

            new_msg = InputMediaVideo(
                base.get_tempfile_url(video, "video/mp4"),
                caption=text,
                thumb=base.get_tempfile_url(thumb, "image/jpeg"),
            )
            self.db.set(
                f"{self.id}:pgn",
                gzip.compress(
                    parsers.PGNParser.encode(
                        self.moves,
                        white_name=self.db.get_name(self.player1),
                        black_name=self.db.get_name(self.player2),
                        date=self.date,
                        result=self.state,
                    ).encode()
                ),
                ex=3600 * 48,
            )
            if isinstance(self, GroupMatch):
                self.msg = cast(Union[Message, base.InlineMessage], self.msg.edit_media(
                    media=new_msg,
                    reply_markup=self._keyboard(
                        [
                            {
                                "text": base.langtable[lang_code]["download-pgn"],
                                "data": ["DOWNLOAD", self.id],
                            }
                        ],
                        handler_id="MAIN",
                    ),
                ))
            elif isinstance(self, PMMatch):
                if self.player_msg:
                    self.player_msg = cast(Message, self.player_msg.edit_media(
                        media=new_msg,
                        reply_markup=self._keyboard(
                            [
                                {
                                    "text": base.langtable[lang_code]["download-pgn"],
                                    "data": ["DOWNLOAD", self.id],
                                }
                            ],
                            handler_id="MAIN",
                        ),
                    ))
                elif self.opponent_msg:
                    self.opponent_msg = cast(Message, self.opponent_msg.edit_media(
                        media=new_msg,
                        reply_markup=self._keyboard(
                            [
                                {
                                    "text": base.langtable[lang_code]["download-pgn"],
                                    "data": ["DOWNLOAD", self.id],
                                }
                            ],
                            handler_id="MAIN",
                        ),
                    ))
                else:
                    raise ValueError("shit")
            else:
                raise ValueError("shit")
        except Exception as exc:
            assert self.dispatcher is not None
            self.dispatcher.dispatch_error(None, exc)
            text += "\n\n" + base.langtable[lang_code]["visualization-error"]
            if isinstance(self, GroupMatch):
                self.msg = cast(Union[Message, base.InlineMessage], self.msg.edit_caption(caption=text))
            elif isinstance(self, PMMatch):
                if self.player_msg:
                    self.player_msg = cast(Message, self.player_msg.edit_caption(caption=text))
                if self.opponent_msg:
                    self.opponent_msg = cast(Message, self.opponent_msg.edit_caption(caption=text))

        if not isinstance(self, AIMatch):
            if self.state in (
                core.GameState.BLACK_CHECKMATED,
                core.GameState.BLACK_RESIGNED,
            ):
                base.set_result(self.id, {self.player1: True, self.player2: False})
            elif self.state in (
                core.GameState.WHITE_CHECKMATED,
                core.GameState.WHITE_RESIGNED,
            ):
                base.set_result(self.id, {self.player1: False, self.player2: True})
            elif self.state == core.GameState.ABORTED:
                base.set_result(self.id, {})
            else:
                base.set_result(self.id, {self.player1: True, self.player2: True})
        else:
            base.set_result(self.id, {})

        logging.info("Match result cached")

    def send_analysis_video(self, lang_code: str) -> None:
        if self.state in (
            core.GameState.WHITE_CHECKMATED,
            core.GameState.WHITE_RESIGNED,
            core.GameState.BLACK_CHECKMATED,
            core.GameState.BLACK_RESIGNED,
        ):
            player, opponent = self.players
            title_text = base.langtable[lang_code][
                self.state.name.lower().replace("_", "-")
            ].format(name=self.db.get_name(player))
            text = "\n".join(
                [
                    title_text,
                    base.langtable[lang_code]["winner"].format(
                        name=self.db.get_name(opponent)
                    ),
                    base.langtable[lang_code]["n-moves"].format(
                        n=self.cur_board.turn - 1
                    ),
                ]
            )
        else:
            title_text = base.langtable[lang_code][
                self.state.name.lower().replace("_", "-")
            ]
            text = "\n".join(
                [
                    title_text,
                    base.langtable[lang_code]["is-draw"],
                    base.langtable[lang_code]["n-moves"].format(
                        n=self.cur_board.turn - 1
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

        last_boardimg = InputMediaPhoto(
            base.get_tempfile_url(
                media.board_image(
                    lang_code,
                    moves=self.moves,
                    startpos=self.startpos,
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
            cast(Union[Message, base.InlineMessage], self.msg)
            self.msg = cast(Union[Message, base.InlineMessage], self.msg.edit_media(media=last_boardimg))
        elif hasattr(self, "player_msg"):
            if self.player_msg:
                self.player_msg = cast(Message, self.player_msg.edit_media(last_boardimg))
            if self.opponent_msg:
                self.opponent_msg = cast(Message, self.opponent_msg.edit_media(last_boardimg))
        eval_thread.start()

    def init_turn(self, move: core.Move = None) -> None:
        logging.debug("Move made: " + (move.pgn_encode() if move else ""))
        if move is not None:
            self.moves.append(move)
        self.state = self.get_state()


class GroupMatch(BaseMatch):
    def __init__(
        self,
        player1: User,
        player2: User,
        msg: Union[Message, base.InlineMessage],
        **kwargs,
    ):
        self.player1: User = player1
        self.player2: User = player2
        self.msg: Union[Message, base.InlineMessage] = msg
        super().__init__(**kwargs)

    def __bytes__(self) -> bytes:
        return parsers.CGNParser.encode(
            self.moves,
            white_name=self.db.get_name(self.player1),
            black_name=self.db.get_name(self.player2),
            date=self.date,
            result=self.state,
            headers={
                "WUID": hex(self.player1.id)[2:],
                "BUID": hex(self.player2.id)[2:],
                "CID": hex(self.msg.chat_id)[2:] if not isinstance(self.msg, base.InlineMessage) else "0",
                "MID": hex(self.msg.message_id)[2:]
                if isinstance(self.msg, Message)
                else self.msg.message_id,
                "MSGTXT": self.init_msg_text or "",
                "OPTIONS": json.dumps(self.options),
            },
        )

    @classmethod
    def _from_cgn(cls, obj: parsers.MatchData, dispatcher: Dispatcher, id: str) -> "GroupMatch":
        player1 = User(
            int(obj["headers"]["WUID"], 16),
            obj["white_name"],
            False,
            bot=dispatcher.bot,
        )
        player2 = User(
            int(obj["headers"]["BUID"], 16),
            obj["black_name"],
            False,
            bot=dispatcher.bot,
        )

        if obj["headers"]["CID"] == "0":
            msg = cast(Union[Message, base.InlineMessage], base.InlineMessage(obj["headers"]["MID"], dispatcher.bot))
        else:
            msg = Message(
                int(obj["headers"]["MID"], 16),
                datetime.datetime.strptime(cast(str, obj["date"]), utils.DATE_FORMAT),
                Chat(int(obj["headers"]["CID"], 16), "group"),
                bot=dispatcher.bot,
            )

        res = cls(
            player1=player1,
            player2=player2,
            msg=msg,
            options=json.loads(obj["headers"]["OPTIONS"]),
            dispatcher=dispatcher,
            id=id,
            date=obj["date"],
        )
        res.init_msg_text = obj["headers"]["MSGTXT"]
        res.moves = obj["moves"]
        return res

    def init_turn(self, move: core.Move = None) -> None:
        super().init_turn(move=move)
        player, opponent = self.players
        langtable = base.langtable[player.language_code]

        if self.state not in (core.GameState.NORMAL, core.GameState.CHECK):
            self.send_analysis_video(cast(str, player.language_code))
        else:
            self.init_msg_text = langtable["curr-move"].format(n=self.cur_board.turn)

            self.init_msg_text += "\n"
            if self.state == core.GameState.CHECK:
                self.init_msg_text += langtable["opponent-in-check"].format(
                    name=self.db.get_name(player)
                ) + "\n"

            self.init_msg_text += langtable["opponent-to-move"].format(
                name=self.db.get_name(player)
            )
            if self.draw_offered:
                self.init_msg_text += "\n" + langtable["player-to-accept-draw"]
                keyboard = [
                    {"text": langtable["decline-button"], "data": ["TURN"]},
                    {"text": langtable["accept-button"], "data": ["ACCEPT_DRAW"]}
                ]
            else:
                self.init_msg_text += "\n" + langtable["player-to-move"]

                keyboard = [
                    {"text": langtable["other-actions-button"], "data": ["OTHER"]}
                ]
                for piece in self.pieces[0]:
                    if piece.get_moves():
                        keyboard.append(
                            {
                                "text": langtable["piece-desc"].format(
                                    piece=core.PGNSYMBOLS["emoji"][type(piece).__name__]
                                    + langtable[type(piece).__name__.lower()],
                                    pos=str(piece.pos),
                                ),
                                "data": ["CHOOSE_PIECE", str(piece.pos)],
                            }
                        )

            self.msg = cast(Union[Message, base.InlineMessage], self.msg.edit_media(
                media=InputMediaPhoto(
                    base.get_tempfile_url(
                        media.board_image(
                            player.language_code,
                            moves=self.moves,
                            startpos=self.startpos,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        "image/jpeg",
                    ),
                    caption=self.init_msg_text,
                ),
                reply_markup=self._keyboard(
                    cast(list[_CallbackDataInput], keyboard), expected_uid=player.id, head_item=True
                ),
            ))

    def handle_input(
        self, command: str, args: list[str]
    ) -> Optional[tuple[str, bool]]:
        try:
            player, opponent = self.players
            langtable = base.langtable[player.language_code]
            allies, _ = self.pieces

            if command == "OTHER":
                new_text = self.init_msg_text.split("\n")
                new_text[-1] = langtable["player-to-choose-action"]
                self.msg = cast(Union[Message, base.InlineMessage], self.msg.edit_caption(
                    caption="\n".join(new_text),
                    reply_markup=self._keyboard(
                        [
                            {"text": langtable["back-button"], "data": ["TURN"]},
                            {
                                "text": langtable["resign-button"],
                                "data": ["RESIGN"],
                            },
                            {
                                "text": langtable["draw-offer-button"],
                                "data": ["OFFER_DRAW"],
                            },
                        ],
                        expected_uid=player.id,
                        head_item=True
                    ),
                ))

            elif command == "TURN":
                if self.draw_offered:
                    init_msg_text = self.init_msg_text.split("\n")
                    init_msg_text[-1] = langtable["player-to-move"]
                    self.init_msg_text = "\n".join(init_msg_text)
                piece_buttons = [
                    {"text": langtable["other-actions-button"], "data": ["OTHER"]}
                ]
                for piece in allies:
                    if piece.get_moves():
                        piece_buttons.append(
                            {
                                "text": langtable["piece-desc"].format(
                                    piece=core.PGNSYMBOLS["emoji"][type(piece).__name__]
                                    + langtable[type(piece).__name__.lower()],
                                    pos=str(piece.pos),
                                ),
                                "data": ["CHOOSE_PIECE", str(piece.pos)],
                            }
                        )

                self.msg = cast(Union[Message, base.InlineMessage], self.msg.edit_media(
                    media=InputMediaPhoto(
                        base.get_tempfile_url(
                            media.board_image(
                                player.language_code,
                                moves=self.moves,
                                startpos=self.startpos,
                                player1_name=self.db.get_name(self.player1),
                                player2_name=self.db.get_name(self.player2),
                            ),
                            "image/jpeg",
                        ),
                        caption=self.init_msg_text,
                    ),
                    reply_markup=self._keyboard(
                        cast(list[_CallbackDataInput], piece_buttons), expected_uid=player.id, head_item=True
                    ),
                ))
                if self.draw_offered and len(args) == 0:
                    self.draw_offered = False
                    return (langtable["draw-offer-declined"], False)

            elif command == "RESIGN":
                self.state = (
                    core.GameState.WHITE_RESIGNED
                    if len(self.moves) % 2 == 0
                    else core.GameState.BLACK_RESIGNED
                )
                self.send_analysis_video(cast(str, opponent.language_code))

            elif command == "ACCEPT_DRAW":
                self.state = core.GameState.DRAW
                self.send_analysis_video(cast(str, player.language_code))

            elif command == "OFFER_DRAW":
                if self.draw_offered:
                    return (langtable["draw-already-offered"], True)
                elif opponent.is_bot:
                    return (langtable["draw-offer-to-bot"], True)
                else:
                    self.draw_offered = True
                    self.handle_input("TURN", ["1"])
                    return (langtable["draw-offered"], True)

            elif command == "CHOOSE_PIECE":
                pos = utils.BoardPoint(args[0])
                dest_buttons = [{"text": langtable["back-button"], "data": ["TURN"]}]
                moves = self.cur_board[pos].get_moves()
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
                new_text[-1] = langtable["player-to-choose-dest"]

                self.msg = cast(Union[Message, base.InlineMessage], self.msg.edit_media(
                    media=InputMediaPhoto(
                        base.get_tempfile_url(
                            media.board_image(
                                player.language_code,
                                moves=self.moves,
                                startpos=self.startpos,
                                selected=pos,
                                possible_moves=moves,
                                player1_name=self.db.get_name(self.player1),
                                player2_name=self.db.get_name(self.player2),
                            ),
                            "image/jpeg",
                        ),
                        caption="\n".join(new_text),
                    ),
                    reply_markup=self._keyboard(
                        cast(list[_CallbackDataInput], dest_buttons), expected_uid=player.id, head_item=True
                    ),
                ))

            elif command == "PROMOTION_MENU":
                src = utils.BoardPoint(args[0])
                dst = utils.BoardPoint(args[1])
                move = core.Move.from_piece(
                    self.cur_board[src], dst, new_piece=core.Queen
                )
                pieces = [
                    {
                        "text": langtable["queen"],
                        "data": ["MOVE", move.pgn_encode()],
                    },
                    {
                        "text": langtable["knight"],
                        "data": ["MOVE", move.copy(new_piece=core.Knight).pgn_encode()],
                    },
                    {
                        "text": langtable["bishop"],
                        "data": ["MOVE", move.copy(new_piece=core.Bishop).pgn_encode()],
                    },
                    {
                        "text": langtable["rook"],
                        "data": ["MOVE", move.copy(new_piece=core.Rook).pgn_encode()],
                    },
                ]
                new_text = self.init_msg_text.split("\n")
                new_text[-1] = langtable["player-to-promote"]

                self.msg = cast(Union[Message, base.InlineMessage], self.msg.edit_media(
                    media=InputMediaPhoto(
                        base.get_tempfile_url(
                            media.board_image(
                                player.language_code,
                                moves=self.moves,
                                startpos=self.startpos,
                                selected=src,
                                possible_moves=[move],
                                player1_name=self.db.get_name(self.player1),
                                player2_name=self.db.get_name(self.player2),
                            ),
                            "image/jpeg",
                        ),
                        caption="\n".join(new_text),
                    ),
                    reply_markup=self._keyboard(cast(list[_CallbackDataInput], pieces), expected_uid=player.id),
                ))

            elif command == "MOVE":
                self.init_turn(move=core.Move.from_pgn(args[0], self.cur_board))
        except Exception as exc:
            assert self.dispatcher is not None
            self.dispatcher.dispatch_error(None, exc)
            base.set_result(self.id, {})
            self.msg.edit_caption(
                caption=base.langtable[self.player1.language_code]["logic-error"]
            )

        return None


class PMMatch(BaseMatch):
    def __init__(
        self,
        player1: User,
        player2: User,
        msg1: Optional[Message],
        msg2: Optional[Message],
        **kwargs,
    ):
        self.player1: User = player1
        self.player2: User = player2
        self.msg1: Optional[Message] = msg1
        self.msg2: Optional[Message] = msg2
        super().__init__(**kwargs)

    def __bytes__(self) -> bytes:
        return parsers.CGNParser.encode(
            self.moves,
            white_name=self.db.get_name(self.player1),
            black_name=self.db.get_name(self.player2),
            date=self.date,
            result=self.state,
            headers={
                "WUID": "BOT" if self.player1.is_bot else hex(self.player1.id)[2:],
                "BUID": "BOT" if self.player2.is_bot else hex(self.player2.id)[2:],
                "CID1": hex(self.msg1.chat_id)[2:] if self.msg1 else "0",
                "CID2": hex(self.msg2.chat_id)[2:] if self.msg2 else "0",
                "MID1": hex(self.msg1.message_id)[2:] if self.msg1 else "0",
                "MID2": hex(self.msg2.message_id)[2:] if self.msg2 else "0",
                "MSGTXT": self.init_msg_text,
                "OPTIONS": json.dumps(self.options),
            },
        )

    @property   # type: ignore
    def player_msg(self) -> Optional[Message]:   # type: ignore
        return self.msg1 if len(self.moves) % 2 == 0 else self.msg2

    @player_msg.setter
    def player_msg(self, msg: Optional[Message]) -> None:
        if len(self.moves) % 2 == 0:
            self.msg1 = msg
        else:
            self.msg2 = msg

    @property     # type: ignore
    def opponent_msg(self) -> Optional[Message]:    # type: ignore
        return self.msg2 if len(self.moves) % 2 == 0 else self.msg1

    @opponent_msg.setter
    def opponent_msg(self, msg: Message) -> None:
        if len(self.moves) % 2 == 0:
            self.msg2 = msg
        else:
            self.msg1 = msg

    @classmethod
    def _from_cgn(cls, obj: parsers.MatchData, dispatcher: Dispatcher, id: str) -> "PMMatch":
        date = datetime.datetime.strptime(cast(str, obj["date"]), utils.DATE_FORMAT)

        res = cls(
            User(
                int(obj["headers"]["WUID"], 16),
                obj["white_name"],
                False,
                bot=dispatcher.bot,
            )
            if obj["headers"]["WUID"] != "BOT"
            else dispatcher.bot.get_me(),
            User(
                int(obj["headers"]["BUID"], 16),
                obj["black_name"],
                False,
                bot=dispatcher.bot,
            )
            if obj["headers"]["BUID"] != "BOT"
            else dispatcher.bot.get_me(),
            Message(
                int(obj["headers"]["MID1"], 16),
                date,
                Chat(int(obj["headers"]["CID1"], 16), "group"),
                text=obj["headers"]["MSGTXT"],
                bot=dispatcher.bot,
            )
            if obj["headers"]["MID1"] != "0"
            else None,
            Message(
                int(obj["headers"]["MID2"], 16),
                date,
                Chat(int(obj["headers"]["CID2"], 16), "group"),
                bot=dispatcher.bot,
            )
            if obj["headers"]["MID2"] != "0"
            else None,
            options=json.loads(obj["headers"]["OPTIONS"]),
            dispatcher=dispatcher,
            id=id,
            date=obj["date"],
        )
        res.init_msg_text = obj["headers"]["MSGTXT"]
        res.moves = obj["moves"]
        return res

    def init_turn(self, move: core.Move = None):
        super().init_turn(move=move)

        player, opponent = self.players
        player_langtable = base.langtable[player.language_code]
        opponent_langtable = base.langtable[opponent.language_code]

        if self.state not in (core.GameState.NORMAL, core.GameState.CHECK):
            self.send_analysis_video(cast(str, player.language_code))
        else:
            self.init_msg_text = player_langtable["curr-move"].format(
                n=self.cur_board.turn
            )
            opponent_msg = opponent_langtable["curr-move"].format(
                n=self.cur_board.turn
            )

            self.init_msg_text += "\n"
            opponent_msg += "\n"
            if self.state == core.GameState.CHECK:
                self.init_msg_text += player_langtable["player-in-check"].format()
                opponent_msg += opponent_langtable["opponent-in-check"].format(
                    name=self.db.get_name(player)
                )

            opponent_msg += opponent_langtable["opponent-to-move"].format(
                name=self.db.get_name(player)
            )
            if self.draw_offered:
                self.init_msg_text += "\n" + player_langtable["player-to-accept-draw"]
                keyboard = [
                    {"text": player_langtable["decline-button"], "data": ["TURN"]},
                    {"text": player_langtable["accept-button"], "data": ["ACCEPT_DRAW"]}
                ]
            else:
                self.init_msg_text += "\n" + player_langtable["player-to-move"]

                keyboard = [
                    {"text": player_langtable["other-actions-button"], "data": ["OTHER"]}
                ]
                for piece in self.pieces[0]:
                    if piece.get_moves():
                        keyboard.append(
                            {
                                "text": player_langtable["piece-desc"].format(
                                    piece=core.PGNSYMBOLS["emoji"][type(piece).__name__]
                                    + player_langtable[type(piece).__name__.lower()],
                                    pos=str(piece.pos),
                                ),
                                "data": ["CHOOSE_PIECE", str(piece.pos)],
                            }
                        )
            img = media.board_image(
                player.language_code,
                moves=self.moves,
                startpos=self.startpos,
                player1_name=self.db.get_name(self.player1),
                player2_name=self.db.get_name(self.player2),
            )
            if self.player_msg:
                self.player_msg = cast(Message, self.player_msg.edit_media(
                    media=InputMediaPhoto(
                        base.get_tempfile_url(img, "image/jpeg"),
                        caption=self.init_msg_text,
                    ),
                    reply_markup=self._keyboard(
                        cast(list[_CallbackDataInput], keyboard), expected_uid=player.id, head_item=True
                    ),
                ))

            if self.opponent_msg:
                self.opponent_msg = cast(Message, self.opponent_msg.edit_media(
                    media=InputMediaPhoto(
                        base.get_tempfile_url(img, "image/jpeg"),
                        caption=opponent_msg,
                    )
                ))

    def handle_input(self, command: str, args: list[str]) -> Optional[tuple[str, bool]]:
        try:
            player, opponent = self.players
            langtable = base.langtable[player.language_code]
            allies, _ = self.pieces
            if command == "OTHER":
                new_text = self.init_msg_text.split("\n")
                new_text[-1] = langtable["player-to-choose-action"]
                self.player_msg = cast(Message, cast(Message, self.player_msg).edit_caption(
                    caption="\n".join(new_text),
                    reply_markup=self._keyboard(
                        [
                            {"text": langtable["back-button"], "data": ["TURN"]},
                            {
                                "text": langtable["resign-button"],
                                "data": ["RESIGN"],
                            },
                            {
                                "text": langtable["draw-offer-button"],
                                "data": ["OFFER_DRAW"],
                            },
                        ],
                        expected_uid=player.id,
                        head_item=True
                    ),
                ))

            if command == "TURN":
                if self.draw_offered:
                        init_msg_text = self.init_msg_text.split("\n")
                        init_msg_text[-1] = langtable["player-to-move"]
                        self.init_msg_text = "\n".join(init_msg_text)
                piece_buttons = [
                    {"text": langtable["other-actions-button"], "data": ["OTHER"]}
                ]
                for piece in allies:
                    if piece.get_moves():
                        piece_buttons.append(
                            {
                                "text": langtable["piece-desc"].format(
                                    piece=core.PGNSYMBOLS["emoji"][type(piece).__name__]
                                    + langtable[type(piece).__name__.lower()],
                                    pos=str(piece.pos),
                                ),
                                "data": ["CHOOSE_PIECE", str(piece.pos)],
                            }
                        )

                self.player_msg = cast(Message, cast(Message, self.player_msg).edit_media(
                    media=InputMediaPhoto(
                        base.get_tempfile_url(
                            media.board_image(
                                player.language_code,
                                moves=self.moves,
                                startpos=self.startpos,
                                player1_name=self.db.get_name(self.player1),
                                player2_name=self.db.get_name(self.player2),
                            ),
                            "image/jpeg",
                        ),
                        caption=self.init_msg_text,
                    ),
                    reply_markup=self._keyboard(
                        cast(list[_CallbackDataInput], piece_buttons), expected_uid=player.id, head_item=True
                    ),
                ))

            elif command == "RESIGN":
                self.state = (
                    core.GameState.WHITE_RESIGNED
                    if len(self.moves) % 2 == 0
                    else core.GameState.BLACK_RESIGNED
                )
                self.send_analysis_video(cast(str, player.language_code))

            elif command == "ACCEPT_DRAW":
                self.state = core.GameState.DRAW
                self.send_analysis_video(cast(str, player.language_code))

            elif command == "OFFER_DRAW":
                if self.draw_offered:
                    return (langtable["draw-already-offered"], True)
                elif opponent.is_bot:
                    return (langtable["draw-offer-to-bot"], True)
                else:
                    self.draw_offered = True
                    self.handle_input("TURN", [])
                    return (langtable["draw-offered"], True)

            elif command == "CHOOSE_PIECE":
                pos = utils.BoardPoint(args[0])
                dest_buttons = [{"text": langtable["back-button"], "data": ["TURN"]}]
                moves = self.cur_board[pos].get_moves()
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

                self.player_msg = cast(Message, cast(Message, self.player_msg).edit_media(
                    media=InputMediaPhoto(
                        base.get_tempfile_url(
                            media.board_image(
                                player.language_code,
                                moves=self.moves,
                                startpos=self.startpos,
                                selected=pos,
                                possible_moves=moves,
                                player1_name=self.db.get_name(self.player1),
                                player2_name=self.db.get_name(self.player2),
                            ),
                            "image/jpeg",
                        ),
                        caption="\n".join(new_text),
                    ),
                    reply_markup=self._keyboard(
                        cast(list[_CallbackDataInput], dest_buttons), expected_uid=player.id, head_item=True
                    ),
                ))

            elif command == "PROMOTION_MENU":
                src = utils.BoardPoint(args[0])
                dst = utils.BoardPoint(args[1])
                move = core.Move.from_piece(self.cur_board[src], dst, new_piece=core.Queen)
                pieces = [
                    {
                        "text": langtable["queen"],
                        "data": ["MOVE", move.pgn_encode()],
                    },
                    {
                        "text": langtable["knight"],
                        "data": ["MOVE", move.copy(new_piece=core.Knight).pgn_encode()],
                    },
                    {
                        "text": langtable["bishop"],
                        "data": ["MOVE", move.copy(new_piece=core.Bishop).pgn_encode()],
                    },
                    {
                        "text": langtable["rook"],
                        "data": ["MOVE", move.copy(new_piece=core.Rook).pgn_encode()],
                    },
                ]

                new_text = self.init_msg_text.split("\n")
                new_text[-1] = langtable["player-to-promote"]

                self.player_msg = cast(Message, cast(Message, self.player_msg).edit_media(
                    media=InputMediaPhoto(
                        base.get_tempfile_url(
                            media.board_image(
                                player.language_code,
                                moves=self.moves,
                                startpos=self.startpos,
                                selected=src,
                                possible_moves=[move],
                                player1_name=self.db.get_name(self.player1),
                                player2_name=self.db.get_name(self.player2),
                            ),
                            "image/jpeg",
                        ),
                        caption="\n".join(new_text),
                    ),
                    reply_markup=self._keyboard(cast(list[_CallbackDataInput], pieces), expected_uid=player.id),
                ))

            elif command == "MOVE":
                return self.init_turn(move=core.Move.from_pgn(args[0], self.cur_board))
        except Exception as exc:
            assert self.dispatcher is not None
            self.dispatcher.dispatch_error(None, exc)
            base.set_result(self.id, {})
            if self.msg1:
                self.msg1.edit_caption(
                    caption=base.langtable[self.player1.language_code]["logic-error"]
                )
            if self.msg2:
                self.msg2.edit_caption(
                    caption=base.langtable[self.player2.language_code]["logic-error"]
                )

        return None



class AIMatch(PMMatch):
    DIFF_PRESETS = {
        "low-diff": (0, 1, 1, 2, 2, 2, 2),
        "mid-diff": (0, 1, 1, 2, 2),
        "high-diff": (0, 1, 2),
        "max-diff": (0,),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = analysis.ChessEngine(self.ENGINE_FILENAME, default_eval_depth=18)
        self.difficulty = self.options["difficulty"]

        self.engine.set_move_probabilities(self.DIFF_PRESETS[self.difficulty])
        self.engine["UCI_Chess960"] = self.is_chess960

    def init_turn(self, move: core.Move = None, **kwargs) -> None:
        if move is None:
            if self.player1.is_bot:
                super().init_turn(self.engine.get_move(self.cur_board))
            elif self.player2.is_bot:
                super().init_turn(**kwargs)
        else:
            super().init_turn(move=move)
            return super().init_turn(self.engine.get_move(self.cur_board))

