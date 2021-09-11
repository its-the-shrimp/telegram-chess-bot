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
from typing import Any, Union, Optional

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


def decode_pgn_moveseq(
    src: str, startpos: core.BoardInfo = core.BoardInfo.from_fen(STARTPOS)
) -> list[core.BoardInfo]:
    states = [startpos]
    *moves, result = src.replace("\n", " ").split()
    for token in moves:
        if not (token[:-1].isdigit() and token[-1] == "."):
            states.append(states[-1] + core.Move.from_pgn(token, states[-1]))

    return states, result


def from_dict(obj: dict[str, Any], match_id: str, bot: Bot) -> "BaseMatch":
    cls = eval(obj["type"] + "Match")
    return cls.from_dict(obj, match_id, bot)


def decode_pgn_seq(pgn: str) -> list[core.BoardInfo]:
    pass


class BaseMatch:
    ENGINE_FILENAME = "./stockfish_14_x64"
    WRONG_PERSON_MSG = "Сейчас не ваш ход!"
    db = None

    def __init__(self, bot=None, id=None):
        self.init_msg_text: Optional[str] = None
        self.last_move: Optional[core.Move] = None
        self.bot: Bot = bot
        self.states: list[core.BoardInfo] = [core.BoardInfo.from_fen(STARTPOS)]
        self.result = "*"
        self.id: str = id if id else "".join(random.choices(IDSAMPLE, k=8))
        self.image_filename: str = f"chess-{self.id}.jpg"
        self.video_filename: str = f"chess-{self.id}.mp4"
        self.game_filename: str = f"telegram-chess-bot-{self.id}.pgn_encode()"

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
        return {"type": "Base", "moves": core.get_pgn_moveseq(core.get_moves(self.states))}

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
            "Site": "https://t.me/real_chessbot", 
            "Date": datetime.datetime.now().strftime("%Y.%m.%d"),
            "Round": 1,
            "White": self.db.get_name(self.player1) if self.db and hasattr(self, "player1") else "?",
            "Black": self.db.get_name(self.player2) if self.db and hasattr(self, "player2") else "?",
            "Result": self.result
        }
        headers = std_headers | headers

        encoded = "\n".join([f'[{k} "{v}"]' for k, v in headers.items()])
        return encoded + "\n\n" + core.get_pgn_moveseq(core.get_moves(self.states), result=self.result)

    #def send_finish_msg(self, msg: Message, text: str):

    def init_turn(self, move: core.Move = None) -> None:
        if move:
            self.states.append(self.states[-1] + move)
        self.last_move = move


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
        new.states = decode_pgn_moveseq(obj["moves"])
        new.init_msg_text = obj["msg_text"]
        new.msg = Message(
            obj["msg_id"],
            datetime.datetime.now().timestamp(),
            Chat(obj["chat_id"], "group", bot=bot),
            bot=bot,
            caption=obj["msg_text"],
        )
        new.turn += 1
        new.empty_halfturns += 1
        new.is_white_turn = not new.is_white_turn

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

    def init_turn(self, move: core.Move = None) -> None:
        super().init_turn(move=move)
        player, opponent = self.players
        state = self.get_state()
        if "draw" in state:
            self.result = "1/2-1/2"
        elif state == "checkmate":
            self.result = "0-1" if self.states[-1].is_white_turn else "1-0"

        if state == "checkmate":
            msg = f"""
Игра окончена: шах и мат!
Победитель: {self.db.get_name(opponent)}
Ходов: {self.states[-1].turn - 1}.
            """
        elif state == "50-move-draw":
            msg = f"""
Игра окончена: за последние 50 ходов не было убито ни одной фигуры и не сдвинуто ни одной пешки.
Ничья!
Ходов: {self.states[-1].turn - 1}
            """
        elif state == "3fold-repetition-draw":
            msg = f"""
Игра окончена: одинаковая позиция доски возникла 3-ий раз подряд.
Ничья!
Ходов: {self.states[-1].turn - 1}
            """
        elif state == "stalemate-draw":
            msg = f"""
Игра окончена: у {"белых" if self.states[-1].is_white_turn else "черных"} не осталось ходов, но их король не под шахом.
Ничья!
Ходов: {self.states[-1].turn - 1}
            """
        elif state == "insufficient-material-draw":
            msg = f"""
Игра окончена: у обеих сторон недостаточно фигур, чтобы поставить мат.
Ничья!
Ходов: {self.states[-1].turn - 1}
            """
        else:
            msg = f"Ход {self.states[-1].turn}"
            if move:
                msg += f"\n{move.piece.name}{' -> '+move.new_piece.name if move.is_promotion else ''}"
                msg += f": {encode_pos(move.src)} -> {encode_pos(move.dst)}"
                if move.is_castling:
                    msg += f" ({'Короткая' if move.rook_src.column == 7 else 'Длинная'} рокировка)"
                if move.is_capturing:
                    msg += f"\n{move.killed.name} на {encode_pos(move.dst)} убит"
                    msg += f"{'а' if move.killed.name in ['Пешка', 'Ладья'] else ''}!"
                else:
                    msg += "\n"
            else:
                msg += "\n\n"

            if state == "check":
                msg += "\nИгроку поставлен шах!"
            else:
                msg += "\n"

            msg += f"\nХодит { self.db.get_name(player) }; выберите действие:"

        if self.result != "*":
            video, thumb = media.board_video(self)
            self.msg.edit_media(
                media=InputMediaVideo(
                    video,
                    caption=msg,
                    filename=self.video_filename,
                    thumb=thumb,
                )
            )
        else:
            keyboard = self._keyboard(
                [
                    {"text": "Ходить", "data": ["TURN"]},
                    {"text": "Сдаться", "data": ["SURRENDER"]},
                ],
                player.id,
            )
            self.init_msg_text = msg
            img = media.board_image(
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

    def handle_input(self, args: list[Union[str, int]]) -> None:
        player, opponent = self.players
        allies, _ = self.pieces
        if args[0] == "INIT_MSG":
            self.msg = self.msg.edit_caption(
                self.init_msg_text,
                reply_markup=self._keyboard(
                    [
                        {"text": "Ходить", "data": ["TURN"]},
                        {"text": "Сдаться", "data": ["SURRENDER"]},
                    ],
                    player.id,
                ),
            )

        if args[0] == "TURN":
            piece_buttons = [{"text": "Назад", "data": ["INIT_MSG"]}]
            for piece in allies:
                if next(filter(lambda x: x.is_legal(), piece.get_moves()), None):
                    piece_buttons.append(
                        {"text": str(piece), "data": ["CHOOSE_PIECE", piece.pos]}
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = f"Ходит {self.db.get_name(player)}; выберите фигуру:"

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
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
            video, thumb = media.board_video(self)
            self.msg = self.msg.edit_media(
                media=InputMediaVideo(
                    video,
                    caption=f"""
Игра окончена: {self.db.get_name(player)} сдался.
Победитель: {self.db.get_name(opponent)}.
Ходов: {self.states[-1].turn - 1}.
                    """,
                    filename=self.video_filename,
                    thumb=thumb,
                )
            )

        elif args[0] == "CHOOSE_PIECE":
            args[1] = decode_pos(args[1])
            dest_buttons = [{"text": "Назад", "data": ["TURN"]}]
            piece = self.states[-1][args[1]]
            moves = list(filter(self.states[-1].is_legal, piece.get_moves()))
            for move in moves:
                if move.is_promotion:
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
            new_text[
                -1
            ] = f"Ходит {self.db.get_name(player)}; выберите новое место фигуры:"

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
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
            move = core.Move.from_piece(self.states[-1][args[1]], args[2], new_piece="q")
            pieces = [
                {"text": "Ферзь", "data": ["MOVE", move.pgn_encode()]},
                {
                    "text": "Конь",
                    "data": ["MOVE", move.copy(new_piece="n").pgn_encode()],
                },
                {
                    "text": "Слон",
                    "data": ["MOVE", move.copy(new_piece="b").pgn_encode()],
                },
                {
                    "text": "Ладья",
                    "data": ["MOVE", move.copy(new_piece="r").pgn_encode()],
                },
            ]
            new_text = self.init_msg_text.split("\n")
            new_text[
                -1
            ] = f"Ходит {self.db.get_name(player)}; выберите фигуру, в которую првератится пешка:"

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
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
            self.init_turn(move=core.Move.from_pgn(args[1], self.states[-1]))


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
        new.states, new.result = decode_pgn_moveseq(obj["moves"])
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
        new.turn += 1
        new.empty_halfturns += 1
        new.is_white_turn = not new.is_white_turn

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

    def _send_analysis_video(self, text: str):
        analyser = analysis.ChessEngine(BaseMatch.ENGINE_FILENAME)
        video, thumb = media.board_video(self, analyser=analyser)
        new_msg = InputMediaVideo(
            video, caption=text, filename=self.video_filename, thumb=thumb
        )
        self.db.set(f"{self.id}:pgn", gzip.compress(self.pgn_encode().encode()), ex=3600*48)

        self.player_msg = self.player_msg.edit_media(
            media=new_msg,
            reply_markup=self._keyboard(
                [{"text": "Скачать партию", "data": ["DOWNLOAD", self.id]}],
                expected_uid=self.players[0].id,
                handler_id="MAIN"
            )
        )
        if self.opponent_msg:
            self.opponent_msg = self.opponent_msg.edit_media(
                media=new_msg,
                reply_markup=self._keyboard(
                    [{"text": "Скачать партию", "data": ["DOWNLOAD", self.id]}],
                    expected_uid=self.players[1].id,
                    handler_id="MAIN"
                )
            )

    def init_turn(self, move: core.Move = None, call_parent_method: bool = True):
        if call_parent_method:
            super().init_turn(move=move)
        player, opponent = self.players
        player_chatid, opponent_chatid = self.chat_ids
        state = self.get_state()
        if "draw" in state:
            self.result = "1/2-1/2"
        elif state == "checkmate":
            self.result = "0-1" if self.states[-1].is_white_turn else "1-0"

        if state == "checkmate":
            player_text = opponent_text = f"""
Игра окончена: шах и мат!
Победитель: {self.db.get_name(opponent)}
Ходов: {self.states[-1].turn - 1}.
            """
        elif state == "50-move-draw":
            player_text = opponent_text = f"""
Игра окончена: за последние 50 ходов не было убито ни одной фигуры и не сдвинуто ни одной пешки.
Ничья!
Ходов: {self.states[-1].turn - 1}
            """
        elif state == "3fold-repetition-draw":
            player_text = opponent_text = f"""
Игра окончена: одинаковая позиция доски возникла 3-ий раз подряд.
Ничья!
Ходов: {self.states[-1].turn - 1}
            """
        elif state == "stalemate-draw":
            player_text = opponent_text = f"""
Игра окончена: у {"белых" if self.states[-1].is_white_turn else "черных"} не осталось ходов, но их король не под шахом.
Ничья!
Ходов: {self.states[-1].turn - 1}
            """
        else:
            player_text = f"Ход {self.states[-1].turn}"
            if move:
                player_text += f"\n{move.piece.name}{' -> '+move.new_piece.name if move.is_promotion else ''}"
                player_text += f": {encode_pos(move.src)} -> {encode_pos(move.dst)}"
                if move.is_castling:
                    player_text += f' ({"Короткая" if  move.rook_src.column == 7 else "Длинная"} рокировка)'
                if move.is_capturing:
                    player_text += f"\n{move.killed.name} на {encode_pos(move.dst)} игрока {self.db.get_name(player)} убит"
                    player_text += (
                        f"{'а' if move.killed.name in ['Пешка', 'Ладья'] else ''}!"
                    )
                else:
                    player_text += "\n"
            else:
                player_text += "\n\n"

            if state == "check":
                player_text += f"\nИгроку {self.db.get_name(player)} поставлен шах!"
            else:
                player_text += "\n"

            opponent_text = player_text

            player_text += "\nВыберите действие:"
            opponent_text += f"\nХодит {self.db.get_name(player)}"

        if self.result != "*":
            eval_thread = threading.Thread(target=self._send_analysis_video, args=(player_text,))
            eval_thread.start()
            last_boardimg = InputMediaPhoto(media.board_image(
                    self.states,
                    player1_name=self.db.get_name(self.player1),
                    player2_name=self.db.get_name(self.player2),
                ),
                caption="Игра окончена, проводится анализ партии, пожалуйста подождите...",
                filename=self.image_filename,
            )
            self.player_msg = self.player_msg.edit_media(last_boardimg)
            if self.opponent_msg:
                self.player_msg = self.player_msg.edit_media(last_boardimg)
        else:
            self.init_msg_text = player_text
            keyboard = self._keyboard(
                [
                    {"text": "Ходить", "data": ["TURN"]},
                    {"text": "Сдаться", "data": ["SURRENDER"]},
                ],
                player.id,
            )
            if self.player_msg:
                self.player_msg = self.player_msg.edit_media(
                    media=InputMediaPhoto(
                        media.board_image(
                            self.states,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        caption=player_text,
                        filename=self.image_filename,
                    ),
                    reply_markup=keyboard,
                )
            elif player_chatid:
                self.player_msg = (
                    self.bot.send_photo(
                        player_chatid,
                        media.board_image(
                            self.states,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        caption=player_text,
                        filename=self.image_filename,
                        reply_markup=keyboard,
                    )
                    if player_chatid
                    else None
                )

            if opponent_chatid:
                if self.opponent_msg:
                    self.opponent_msg = self.opponent_msg.edit_media(
                        media=InputMediaPhoto(
                            media.board_image(
                                self.states,
                                player1_name=self.db.get_name(self.player1),
                                player2_name=self.db.get_name(self.player2),
                            ),
                            caption=opponent_text,
                            filename=self.image_filename,
                        )
                    )
                else:
                    self.opponent_msg = self.bot.send_photo(
                        opponent_chatid,
                        media.board_image(
                            self.states,
                            player1_name=self.db.get_name(self.player1),
                            player2_name=self.db.get_name(self.player2),
                        ),
                        caption=opponent_text,
                        filename=self.image_filename,
                    )

    def handle_input(self, args):
        player, opponent = self.players
        allies, _ = self.pieces

        if args[0] == "INIT_MSG":
            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        self.states,
                        player1_name=self.db.get_name(self.player1),
                        player2_name=self.db.get_name(self.player2),
                    ),
                    caption=self.init_msg_text,
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(
                    [
                        {"text": "Ходить", "data": ["TURN"]},
                        {"text": "Сдаться", "data": ["SURRENDER"]},
                    ],
                    player.id,
                ),
            )

        if args[0] == "TURN":
            piece_buttons = [{"text": "Назад", "data": ["INIT_MSG"]}]
            for piece in allies:
                if next(
                    filter(lambda x: x.is_legal(), piece.get_moves()),
                    None,
                ):
                    piece_buttons.append(
                        {"text": str(piece), "data": ["CHOOSE_PIECE", piece.pos]}
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = f"Выберите фигуру:"

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
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
            text=f"""
Игра окончена: {self.db.get_name(player)} сдался.
Победитель: {self.db.get_name(opponent)}.
Ходов: {self.states[-1].turn - 1}."""
            eval_thread = threading.Thread(target=self._send_analysis_video, args=(text,))
            eval_thread.start()
            last_boardimg = InputMediaPhoto(media.board_image(
                    self.states,
                    player1_name=self.db.get_name(self.player1),
                    player2_name=self.db.get_name(self.player2),
                ),
                caption="Игра окончена, проводится анализ партии, пожалуйста подождите...",
                filename=self.image_filename,
            )
            self.player_msg = self.player_msg.edit_media(last_boardimg)
            if self.opponent_msg:
                self.player_msg = self.player_msg.edit_media(last_boardimg)

        elif args[0] == "CHOOSE_PIECE":
            args[1] = decode_pos(args[1])
            dest_buttons = [{"text": "Назад", "data": ["TURN"]}]
            piece = self.states[-1][args[1]]
            moves = list(filter(lambda x: x.is_legal(), piece.get_moves()))
            for move in moves:
                if move.is_promotion:
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
            new_text[-1] = f"Выберите новое место фигуры:"

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
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
            move = core.Move.from_piece(self.states[-1][args[1]], args[2], new_piece="q")
            pieces = [
                {"text": "Ферзь", "data": ["MOVE", move.pgn_encode()]},
                {
                    "text": "Конь",
                    "data": ["MOVE", move.copy(new_piece="n").pgn_encode()],
                },
                {
                    "text": "Слон",
                    "data": ["MOVE", move.copy(new_piece="b").pgn_encode()],
                },
                {
                    "text": "Ладья",
                    "data": ["MOVE", move.copy(new_piece="r").pgn_encode()],
                },
            ]

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = f"Выберите фигуру, в которую првератится пешка:"

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
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
            return self.init_turn(move=core.Move.from_pgn(args[1], self.states[-1]))


class AIMatch(PMMatch):
    PRESET_1 = (0, 1, 1, 2, 2, 2, 2)
    PRESET_2 = (0, 1, 1, 2, 2)
    PRESET_3 = (0, 1, 2)
    PRESET_4 = (0,)
    EVAL_DEPTH = 18

    def __init__(self, player: User, chat_id: int, player2: User = None, **kwargs):
        ai_player = player2 if player2 else kwargs["bot"].get_me()
        self.ai_rating: int = None
        super().__init__(player, ai_player, chat_id, 0, **kwargs)
        self.engine = analysis.ChessEngine(self.ENGINE_FILENAME)

    @classmethod
    def from_dict(cls, obj: JSON, match_id: int, bot=Bot) -> "AIMatch":
        logging.debug(f"Constructing {cls.__name__} object: {obj}")
        player = User.de_json(obj["player1"] | {"is_bot": False}, bot)
        new = cls(player, obj["chat_id1"], bot=bot, id=match_id)
        new.states, new.result = decode_pgn_moveseq(obj["moves"])
        new.init_msg_text = obj["msg_text"]
        new.engine.set_move_probabilities(getattr(new, f"PRESET_{obj['diff']}"))
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
        del res["player2"]
        del res["msg_id2"]
        del res["chat_id2"]
        res.update({"diff": self.difficulty, "type": "AI"})
        return res

    def init_turn(self, setup: bool = False, **kwargs) -> None:
        if setup:
            self.msg1 = self.bot.send_photo(
                self.chat_id1,
                media.board_image(self.states),
                caption="Выберите уровень сложности:",
                filename=self.image_filename,
                reply_markup=self._keyboard(
                    [
                        {"text": "Низкий", "data": ["SKILL_LEVEL", 1]},
                        {"text": "Средний", "data": ["SKILL_LEVEL", 2]},
                        {"text": "Высокий", "data": ["SKILL_LEVEL", 3]},
                        {"text": "Неограниченный", "data": ["SKILL_LEVEL", 4]},
                    ],
                    self.player1.id,
                ),
            )

        else:
            super().init_turn(**kwargs)
            return super().init_turn(self.engine.get_move(self.states[-1], self.EVAL_DEPTH))

    def handle_input(self, args):
        if args[0] == "SKILL_LEVEL":
            self.difficulty = args[1]
            self.engine.set_move_probabilities(getattr(self, f"PRESET_{args[1]}"))
            return super().init_turn()
        else:
            return super().handle_input(args)
