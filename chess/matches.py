import itertools
import random
import subprocess
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
import time
from .base import *
from . import pieces, board, media
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


def from_dict(obj: dict[str, Any], match_id: str, bot: Bot) -> "BaseMatch":
    cls = eval(obj["type"] + "Match")
    return cls.from_dict(obj, match_id, bot)


def decode_pgn_seq(pgn: str) -> list[board.BoardInfo]:
    pass


class BaseMatch:
    WRONG_PERSON_MSG = "Сейчас не ваш ход!"
    db = None

    def __init__(self, bot=None, id=None):
        self.init_msg_text: Optional[str] = None
        self.last_move: Optional[board.Move] = None
        self.bot: Bot = bot
        self.states: list[board.BoardInfo] = [board.BoardInfo.from_fen(STARTPOS)]
        self.finished: bool = False
        self.id: str = id if id else "".join(random.choices(IDSAMPLE, k=8))
        self.image_filename: str = f"chess-{self.id}.jpg"
        self.video_filename: str = f"chess-{self.id}.mp4"
        self.game_filename: str = f"telegram-chess-bot-{self.id}.pgn"

    def _keyboard(
        self,
        seq: list[dict[str, Union[str, BoardPoint]]],
        expected_uid: int,
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
                    callback_data=f"{expected_uid if expected_uid else ''}\n{self.id}\n{'#'.join(data)}",
                )
            )

        if res:
            return InlineKeyboardMarkup(_group_items(res, 2, head_item=head_item))
        else:
            return None

    def get_moves(self) -> list[board.Move]:
        res = [None]
        for index in range(1, len(self.states)):
            res.append(self.states[index] - self.states[index - 1])

        return res

    def to_dict(self) -> JSON:
        return {"type": "Base", "states": [board.fen for board in self.states]}

    @property
    def pieces(self) -> tuple[list[pieces.BasePiece], list[pieces.BasePiece]]:
        return (
            (self.states[-1].whites, self.states[-1].blacks)
            if self.states[-1].is_white_turn
            else (self.states[-1].blacks, self.states[-1].whites)
        )

    def get_state(self) -> Optional[str]:
        cur_king = self.states[-1][pieces.King, self.states[-1].is_white_turn][0]
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

        return "normal"

    def init_turn(self, move: board.Move = None) -> None:
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
        new.states = [board.BoardInfo.from_fen(fen) for fen in obj["states"]]
        new.init_msg_text = obj["msg_text"]
        new.msg = Message(
            obj["msg_id"],
            time.monotonic,
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

    def init_turn(self, move: board.Move = None) -> None:
        super().init_turn(move=move)
        player, opponent = self.players
        state = self.get_state()
        self.finished = state != "normal"

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
        else:
            msg = f"Ход {self.states[-1].turn}"
            if move:
                msg += f"\n{move.piece.name}{' -> '+move.new_piece.name if move.is_promotion else ''}"
                msg += f": {encode_pos(move.src)} -> {encode_pos(move.dst)}"
                if move.is_castling:
                    msg += f" ({'Короткая' if move.rook_src.column == 7 else 'Длинная'} рокировка)"
                if move.is_killing:
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

        if self.finished:
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
            img = media.board_image(self.states[-1], prev_move=self.last_move)
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
                if next(filter(self.states[-1].is_legal, piece.get_moves()), None):
                    piece_buttons.append(
                        {"text": str(piece), "data": ["CHOOSE_PIECE", piece.pos]}
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = f"Ходит {self.db.get_name(player)}; выберите фигуру:"

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(self.states[-1], prev_move=self.last_move),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(piece_buttons, player.id, head_item=True),
            )

        elif args[0] == "SURRENDER":
            self.finished = True
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
                            "text": "⏫" + encode_pos(move.dst),
                            "data": ["PROMOTION_MENU", args[1], move.dst],
                        }
                    )
                else:
                    dest_buttons.append(
                        {
                            "text": MOVETYPE_MARKERS[move.type] + encode_pos(move.dst),
                            "data": ["MOVE", move.pgn],
                        }
                    )
            new_text = self.init_msg_text.split("\n")
            new_text[
                -1
            ] = f"Ходит {self.db.get_name(player)}; выберите новое место фигуры:"

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        self.states[-1],
                        prev_move=self.last_move,
                        selected=args[1],
                        possible_moves=moves,
                    ),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(dest_buttons, player.id, head_item=True),
            )

        elif args[0] == "PROMOTION_MENU":
            move = self.states[-1][args[1]].create_move(args[2], new_piece=pieces.Queen)
            pieces = [
                {"text": "Ферзь", "data": ["MOVE", move.pgn]},
                {"text": "Конь", "data": ["MOVE", move.copy(new_piece=pieces.Knight).pgn]},
                {"text": "Слон", "data": ["MOVE", move.copy(new_piece=pieces.Bishop).pgn]},
                {"text": "Ладья", "data": ["MOVE", move.copy(new_piece=pieces.Rook).pgn]},
            ]
            new_text = self.init_msg_text.split("\n")
            new_text[
                -1
            ] = f"Ходит {self.db.get_name(player)}; выберите фигуру, в которую првератится пешка:"

            self.msg = self.msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        self.states[-1],
                        prev_move=self.last_move,
                        selected=args[1],
                        possible_moves=[move],
                    ),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(pieces, player.id),
            )

        elif args[0] == "MOVE":
            self.init_turn(move=board.Move.from_pgn(args[1], self.states[-1]))


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
        new.states = [board.BoardInfo.from_fen(fen) for fen in obj["states"]]
        new.init_msg_text = obj["msg_text"]
        new.msg1 = Message(
            obj["msg_id1"],
            time.monotonic,
            Chat(obj["chat_id1"], "private", bot=bot),
            bot=bot,
            caption=obj["msg_text"],
        )
        new.msg2 = Message(
            obj["msg_id2"],
            time.monotonic,
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

    def init_turn(self, move: board.Move = None, call_parent_method: bool = True):
        if call_parent_method:
            super().init_turn(move=move)
        player, opponent = self.players
        player_chatid, opponent_chatid = self.chat_ids
        state = self.get_state()
        self.finished = state != "normal"

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
                if move.is_killing:
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

        if self.finished:
            video, thumb = media.board_video(self)
            new_msg = InputMediaVideo(
                video, caption=player_text, filename=self.video_filename, thumb=thumb
            )
            self.player_msg = self.player_msg.edit_media(media=new_msg)
            if self.opponent_msg:
                self.opponent_msg = self.opponent_msg.edit_media(media=new_msg)
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
                        media.board_image(self.states[-1], prev_move=self.last_move),
                        caption=player_text,
                        filename=self.image_filename,
                    ),
                    reply_markup=keyboard,
                )
            else:
                self.player_msg = self.bot.send_photo(
                    player_chatid,
                    media.board_image(self.states[-1], prev_move=self.last_move),
                    caption=player_text,
                    filename=self.image_filename,
                    reply_markup=keyboard,
                )

            if opponent_chatid:
                if self.player_msg:
                    self.opponent_msg = self.opponent_msg.edit_media(
                        media=InputMediaPhoto(
                            media.board_image(self.states[-1], prev_move=self.last_move),
                            caption=opponent_text,
                            filename=self.image_filename,
                        )
                    )
                else:
                    self.opponent_msg = self.bot.send_photo(
                        opponent_chatid,
                        media.board_image(self.states[-1], prev_move=self.last_move),
                        caption=opponent_text,
                        filename=self.image_filename,
                    )

    def handle_input(self, args):
        player, opponent = self.players
        allies, _ = self.pieces

        if args[0] == "INIT_MSG":
            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(self.states[-1], prev_move=self.last_move),
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
                    filter(self.states[-1].is_legal, piece.get_moves()),
                    None,
                ):
                    piece_buttons.append(
                        {"text": str(piece), "data": ["CHOOSE_PIECE", piece.pos]}
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = f"Выберите фигуру:"

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(self.states[-1], prev_move=self.last_move),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(piece_buttons, player.id, head_item=True),
            )

        elif args[0] == "SURRENDER":
            self.finished = True
            video, thumb = media.board_video(self)
            for msg in [self.msg1, self.msg2]:
                if msg:
                    msg.edit_media(
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
                            "data": ["MOVE", move.pgn],
                        }
                    )

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = f"Выберите новое место фигуры:"

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        self.states[-1],
                        prev_move=self.last_move,
                        selected=args[1],
                        possible_moves=moves,
                    ),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(dest_buttons, player.id, head_item=True),
            )

        elif args[0] == "PROMOTION_MENU":
            args[1] = decode_pos(args[1])
            args[2] = decode_pos(args[2])
            move = self.states[-1][args[1]].create_move(args[2], new_piece=pieces.Queen)
            pieces = [
                {"text": "Ферзь", "data": ["MOVE", move.pgn]},
                {"text": "Конь", "data": ["MOVE", move.copy(new_piece=pieces.Knight).pgn]},
                {"text": "Слон", "data": ["MOVE", move.copy(new_piece=pieces.Bishop).pgn]},
                {"text": "Ладья", "data": ["MOVE", move.copy(new_piece=pieces.Rook).pgn]},
            ]

            new_text = self.init_msg_text.split("\n")
            new_text[-1] = f"Выберите фигуру, в которую првератится пешка:"

            self.player_msg = self.player_msg.edit_media(
                media=InputMediaPhoto(
                    media.board_image(
                        self.states[-1],
                        prev_move=self.last_move,
                        selected=args[1],
                        possible_moves=[move],
                    ),
                    caption="\n".join(new_text),
                    filename=self.image_filename,
                ),
                reply_markup=self._keyboard(pieces, player.id),
            )

        elif args[0] == "MOVE":
            return self.init_turn(move=board.Move.from_pgn(args[1], self.states[-1]))


class AIMatch(PMMatch):
    engine_filename = "./stockfish_14_x64"

    def __init__(self, player: User, chat_id: int, player2: User = None, **kwargs):
        ai_player = player2 if player2 else kwargs["bot"].get_me()
        self.ai_rating: int = None
        super().__init__(player, ai_player, chat_id, 0, **kwargs)
        self.engine_api = subprocess.Popen(
            self.engine_filename,
            bufsize=1,
            universal_newlines=True,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        self.engine_api.stdout.readline()
        self.engine_api.stdin.write("setoption name UCI_LimitStrength value true\n")

    @classmethod
    def from_dict(cls, obj: JSON, match_id: int, bot=Bot) -> "AIMatch":
        logging.debug(f"Constructing {cls.__name__} object: {obj}")
        player = User.de_json(obj["player1"] | {"is_bot": False}, bot)
        new = cls(player, obj["chat_id1"], bot=bot, id=match_id)
        new.states = [board.BoardInfo.from_fen(fen) for fen in obj["states"]]
        new.init_msg_text = obj["msg_text"]
        new.set_elo(obj["ai_rating"])
        new.msg1 = Message(
            obj["msg_id1"],
            time.monotonic,
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
        res.update({"ai_rating": self.ai_rating, "type": "AI"})
        return res

    def set_elo(self, value: int) -> None:
        self.ai_rating = value
        self.engine_api.stdin.write(f"setoption name UCI_Elo value {value}\n")

    def init_turn(self, setup: bool = False, **kwargs) -> None:
        if setup:
            self.msg1 = self.bot.send_photo(
                self.chat_id1,
                media.board_image(self.states[-1], prev_move=self.last_move),
                caption="Выберите уровень сложности:",
                filename=self.image_filename,
                reply_markup=self._keyboard(
                    [
                        {"text": "Низкий", "data": ["SKILL_LEVEL", "1350"]},
                        {"text": "Средний", "data": ["SKILL_LEVEL", "1850"]},
                        {"text": "Высокий", "data": ["SKILL_LEVEL", "2350"]},
                        {"text": "Легендарный", "data": ["SKILL_LEVEL", "2850"]},
                    ],
                    self.player1.id,
                ),
            )

        else:
            turn_info = BaseMatch.init_turn(self, **kwargs)
            if self.finished:
                return super().init_turn(self, call_parent_method=False, **kwargs)

            self.engine_api.stdin.write(f"position fen {self.states[-1].fen}\n")
            self.engine_api.stdin.write(f"go depth 2\n")
            for line in self.engine_api.stdout:
                if "bestmove" in line:
                    turn = line.split(" ")[1].strip("\n")
                    break
            return super().init_turn(
                move=self.states[-1][decode_pos(turn[:2])].create_move(
                    decode_pos(turn[2:4]),
                    new_piece=getattr(pieces, FENSYMBOLS[turn[-1]]) if len(turn) == 5 else None,
                ),
            )

    def handle_input(self, args):
        if args[0] == "SKILL_LEVEL":
            self.set_elo(args[1])
            return super().init_turn()
        else:
            return super().handle_input(args)
