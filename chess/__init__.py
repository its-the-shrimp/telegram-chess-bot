from .base import *

from . import media, analysis
from .utils import BoardPoint, STARTPOS
from .core import (
    Move,
    BoardInfo,
    BasePiece,
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    MoveEval,
    GameState,
)
from .matches import (
    BaseMatch,
    GroupMatch,
    PMMatch,
    AIMatch,
    from_bytes,
    get_pgn_file,
)
from .parsers import PGNParser, CGNParser


def init(is_debug: bool):
    BaseMatch.ENGINE_FILENAME = "./stockfish" if is_debug else "./stockfish_14_x64"
    BaseMatch.db = get_database()


def test_match(update: Update, context: BoardGameContext):
    if context.args:
        assert update.effective_message and update.effective_user and update.effective_chat
        options = context.menu.defaults
        options["pos"] = " ".join(context.args[:-1])
        msg = update.effective_chat.send_photo(
            get_file_url(INVITE_IMAGE),
            caption=context.langtable["main:starting-match"]
        )
        msg2 = None
        if context.args[-1] == "group":
            new: BaseMatch = GroupMatch(
                update.effective_user,
                update.effective_user,
                msg,
                dispatcher=context.dispatcher,
                options=options,
            )
            context.bot_data["matches"][new.id] = new
        elif context.args[-1] == "pm":
            msg2 = update.effective_chat.send_photo(
                get_file_url(INVITE_IMAGE),
                context.langtable["main:starting-match"]
            )
            new = PMMatch(
                update.effective_user,
                update.effective_user,
                msg,
                msg2,
                dispatcher=context.dispatcher,
                options=options
            )
        else:
            new = AIMatch(
                update.effective_user,
                context.bot.get_me(),
                msg,
                msg2,
                dispatcher=context.dispatcher,
                options=options
            )
        try:
            new.init_turn()
        except Exception as exc:
            del context.bot_data["matches"][new.id]
            msg.edit_caption(
                context.langtable["main:init-error"],
            )
            if msg2 is not None:
                msg2.edit_caption(
                    context.langtable["main:init-error"]
                )
            context.dispatcher.dispatch_error(update, exc)


OPTIONS = {
    "ruleset": {"values": {"std-chess": None}},
    "mode": {
        "values": {
            "online": lambda obj: obj["ruleset"] != "custom-pos",
            "vsbot": lambda obj: obj["ruleset"] != "fog-of-war",
            "invite": None,
        },
    },
    "timectrl": {
        "values": {"classic": None, "rapid": None, "blitz": None},
        "condition": lambda obj: obj["mode"] != "vsbot",
    },
    "difficulty": {
        "values": {
            "low-diff": None,
            "mid-diff": None,
            "high-diff": None,
            "max-diff": None,
        },
        "condition": lambda obj: obj["mode"] == "vsbot",
    },
}
TEXT_COMMANDS: dict[str, TextCommand] = {"test": test_match}
KEYBOARD_COMMANDS: dict[str, KeyboardCommand] = {"DOWNLOAD": get_pgn_file}
ERROR_IMAGE = "error-404.jpg"
INVITE_IMAGE = "inline-thumb.jpg"
