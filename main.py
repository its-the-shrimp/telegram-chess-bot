import io
import queue
import os, sys
import json
from typing import Iterator, Optional, Callable, Any, Generator, Type, Union
import telegram as tg
from telegram import messageautodeletetimerchanged
import chess as core
import difflib
import logging
import re
import traceback
import flask
import time
import pickle
import redis


INLINE_OPTION_PATTERN = re.compile(r"\w+:\d")
flask_callbacks: list[tuple, dict] = []
tg_handlers: list[tg.ext.Handler] = []

debug_env_path = os.path.join(os.path.dirname(__file__), "debug_env.json")
IS_DEBUG = os.path.exists(debug_env_path)
if IS_DEBUG:
    with open(debug_env_path) as r:
        os.environ.update(json.load(r))


class RedisInterface(redis.Redis):
    bot: tg.Bot

    def _fetch_matches(
        self, decoder: Callable[[bytes], Any], dispatcher: tg.ext.Dispatcher
    ) -> Generator[tuple[str, Any], None, None]:
        for key in self.scan_iter(match="match:*"):
            mid = key.split(b":")[1].decode()
            yield mid, decoder(self.get(key), dispatcher, mid)

    def _flush_matches(self, matches: dict[str, object]) -> None:
        for id, matchobj in matches.items():
            self.set(f"match:{id}", bytes(matchobj))

    def get_pending_message(
        self, pmid: str
    ) -> Optional[tuple[Callable[[tg.Update, "BoardGameContext"], None], tuple, dict]]:
        raw_f = self.get(f"pm:{pmid}:f")
        if int(self.get(f"pm:{pmid}:is-single").decode()) and raw_f:
            self.delete(f"pm:{pmid}:f", f"pm:{pmid}:is-single")
            return pickle.loads(raw_f)

    def get_name(self, user: tg.User) -> str:
        """
        If user has enabled anonymous mode, returns "player(ID of the user)",
        otherwise returns his username, or full name, if no username is available.
        Arguments:
            user: `User` - User whose name needs to bee queried.
        Returns: `str` - Name of the user.
        """
        if self.exists(f"{user.id}:isanon"):
            return f"player{user.id}"
        else:
            return user.name

    def create_invite(self, invite_id: str, user: tg.User, options: dict):
        self.set(
            f"invite:{invite_id}",
            json.dumps({"from_user": user.to_dict(), "options": options}).encode(),
            ex=1800,
        )

    def get_invite(self, invite_id: str) -> Optional[dict]:
        raw = self.get(f"invite:{invite_id}")
        if raw:
            self.delete(f"invite:{invite_id}")
            res = json.loads(raw.decode())
            res["from_user"] = tg.User.de_json(res["from_user"], self.bot)
            return res

    def set_anon_mode(self, user: tg.User, value: bool) -> None:
        if value:
            self.set(f"{user.id}:isanon", b"1")
        else:
            self.delete(f"{user.id}:isanon")

    def get_user_ids(self) -> Generator[int, None, None]:
        for key in self.scan_iter(match="*:lang"):
            key = key.decode()
            yield int(key[: key.find(":")])

    def get_langcodes_stats(self) -> dict[str, int]:
        counter = {}
        for key in self.keys(pattern="*:lang"):
            key = self.get(key).decode()
            if key in counter:
                counter[key] += 1
            else:
                counter[key] = 1

        return counter

    def __getitem__(self, user_id: int) -> Union[dict, bytes]:
        if isinstance(user_id, int):
            return {
                "lang_code": (self.get(f"{user_id}:lang") or b"en").decode(),
                "is_anon": bool(self.exists(f"{user_id}:isanon")),
                "total": int(self.get(f"{user_id}:total") or 0),
                "wins": int(self.get(f"{user_id}:wins") or 0),
            }
        else:
            return super().__getitem__(user_id)

    def __delitem__(self, user_id: int) -> bool:
        if isinstance(user_id, int):
            return bool(
                self.delete(
                    f"{user_id}:lang",
                    f"{user_id}:isanon",
                    f"{user_id}:total",
                    f"{user_id}:wins",
                )
            )
        else:
            return bool(super().__delitem__(user_id))


class OptionValue:
    def __init__(self, value: str, condition: Callable[[dict[str, str]], bool] = None):
        self.value = value
        self.is_available = condition or (lambda _: True)

    def __repr__(self):
        return f"<OptionValue({self.value})>"

    def __eq__(self, other):
        return isinstance(self, other.__class__) and self.value == other.value


class MenuOption:
    @classmethod
    def from_dict(cls, name: str, obj: dict) -> "MenuOption":
        return cls(
            name,
            [OptionValue(k, condition=v) for k, v in obj["values"].items()],
            condition=obj.get("condition"),
        )

    def __init__(
        self,
        name: str,
        values: list[OptionValue],
        condition: Callable[[dict[str, str]], bool] = None,
    ):
        self.name = name
        self.values = values
        self.is_available = condition or (lambda _: True)

    def __repr__(self):
        return f"<MenuOption({self.name}); {len(self.values)} values>"

    def __iter__(self) -> Iterator[OptionValue]:
        return self.values.__iter__()

    def available_values(self, indexes: dict[str, str]) -> list[str]:
        if not self.is_available(indexes):
            return []
        return [i.value for i in self.values if i.is_available(indexes)]


class MenuFormatter:
    @classmethod
    def from_dict(cls, obj: dict) -> "MenuFormatter":
        return cls([MenuOption.from_dict(*args) for args in obj.items()])

    def __init__(self, options: list[MenuOption]):
        self.options: list[MenuOption] = options
        self.defaults: dict[str, int] = {}

        for option in self:
            if not self.defaults:
                self.defaults[option.name] = option.values[0].value
            else:
                self.defaults[option.name] = self.get_default_value(
                    option.name, self.defaults
                )

    def __iter__(self) -> Iterator[MenuOption]:
        return self.options.__iter__()

    def __getitem__(self, key: str) -> MenuOption:
        for option in self:
            if option.name == key:
                return option

        raise KeyError(key)

    def get_value(self, key: str, index: int) -> str:
        lookup_list = [value.value for value in self[key]]
        return lookup_list[index]

    def get_index(self, key: str, value: str) -> int:
        lookup_list = [value.value for value in self[key]]
        return lookup_list.index(value)

    def get_default_value(self, name: str, indexes: dict[str, str]) -> str:
        try:
            return self[name].available_values(indexes)[0]
        except IndexError:
            return None

    def decode(self, keyboard: tg.InlineKeyboardMarkup) -> dict[str, str]:
        res = {}
        for column in keyboard.inline_keyboard[:-1]:
            column_data = core.parse_callback_data(column[1].callback_data)["args"]
            res[column_data[0]] = self.get_value(column_data[0], column_data[1])

        return res

    def encode(
        self, user: tg.User, indexes: dict[str, str] = None
    ) -> tg.InlineKeyboardMarkup:
        res = []
        indexes = indexes or self.defaults

        for option in self:
            if option.is_available(indexes):
                if option.name in indexes:
                    chosen = indexes[option.name]
                else:
                    chosen = self.get_default_value(option.name, indexes)
                    indexes[option.name] = chosen
                res.append(
                    [
                        tg.InlineKeyboardButton(
                            text="◀️",
                            callback_data=core.format_callback_data(
                                "PREV",
                                args=[option.name],
                                expected_uid=user.id,
                                handler_id="MAIN",
                            ),
                        ),
                        tg.InlineKeyboardButton(
                            text=core.langtable[user.language_code][chosen],
                            callback_data=core.format_callback_data(
                                "DESC",
                                args=[option.name, self.get_index(option.name, chosen)],
                                handler_id="MAIN",
                                expected_uid=user.id,
                            ),
                        ),
                        tg.InlineKeyboardButton(
                            text="▶️",
                            callback_data=core.format_callback_data(
                                "NEXT",
                                args=[option.name],
                                handler_id="MAIN",
                                expected_uid=user.id,
                            ),
                        ),
                    ]
                )
            else:
                if option.name in indexes:
                    del indexes[option.name]

        inline_query = self.tg_encode(indexes) if indexes["mode"] == "invite" else None

        res.append(
            [
                tg.InlineKeyboardButton(
                    text=core.langtable[user.language_code]["main:cancel-button"],
                    callback_data=core.format_callback_data(
                        "CANCEL", handler_id="MAIN", expected_uid=user.id
                    ),
                ),
                tg.InlineKeyboardButton(
                    text=core.langtable[user.language_code]["main:play-button"],
                    switch_inline_query=inline_query,
                    callback_data=None
                    if inline_query
                    else core.format_callback_data(
                        "PLAY", handler_id="MAIN", expected_uid=user.id
                    ),
                ),
            ]
        )

        return tg.InlineKeyboardMarkup(res)

    def prettify(self, indexes: dict[str, str], lang_code: str) -> str:
        locale = core.langtable[lang_code]

        return "\n".join(
            [
                locale["main:options-title"],
                ", ".join([locale[i] for i in indexes.values()]),
            ]
        )

    def tg_encode(self, indexes: dict[str, str]) -> str:
        return " ".join([":".join((k, v)) for k, v in indexes.items() if k != "mode"])

    def tg_decode(self, src: str) -> dict[str, str]:
        indexes = dict(token.split(":") for token in src.split(" ") if ":" in token)
        for option in self:
            is_available = option.is_available(indexes)
            if option.name not in indexes and is_available:
                indexes[option.name] = self.get_default_value(option.name, indexes)
            elif option.name in indexes and not is_available:
                del indexes[option.name]

        return indexes


class BoardGameContext(tg.ext.CallbackContext):
    menu: MenuFormatter = MenuFormatter.from_dict(core.OPTIONS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.langtable: dict[str, str] = None
        RedisInterface.bot = self.dispatcher.bot

    @classmethod
    def from_update(
        cls, update: tg.Update, dispatcher: tg.ext.Dispatcher
    ) -> "BoardGameContext":
        self = super().from_update(update, dispatcher)
        if update is not None:
            if update.effective_user is not None:
                self.langtable = core.langtable[update.effective_user.language_code]

            self.db.set(
                f"{update.effective_user.id}:lang",
                update.effective_user.language_code.encode(),
            )

        return self

    @property
    def db(self) -> RedisInterface:
        return self.dispatcher.bot_data["conn"]


class TelegramBotApp(flask.Flask):
    dispatcher: tg.ext.Dispatcher


def get_opponent(queue: list[dict], options: dict) -> Optional[dict]:
    for queuepoint in reversed(queue):
        if options == queuepoint["options"]:
            return queuepoint


def _tg_adapter(f: core.TelegramCallback) -> core.TelegramCallback:
    def decorated(update: tg.Update, context: BoardGameContext):
        try:
            f(update, context)
        except Exception as exc:
            context.dispatcher.dispatch_error(update, exc)

        end = time.monotonic_ns()
        logging.info(
            f"Handled update #{update.update_id} in {(end - context.bot_data['pending_updates'][update.update_id]) // 1000000} ms"
        )
        del context.bot_data["pending_updates"][update.update_id]

    return decorated


def flask_callback(*args, **kwargs):
    def decorator(f: Callable) -> Callable:
        flask_callbacks.append((args, kwargs | {"view_func": f}))
        return f

    return decorator


def tg_callback(handler_type: Type[tg.ext.Handler], *args, **kwargs):
    def decorator(f: core.TelegramCallback) -> core.TelegramCallback:
        def decorated(update: tg.Update, context: BoardGameContext):

            if getattr(update.effective_chat, "type", tg.Chat.PRIVATE) in (
                tg.Chat.PRIVATE,
                tg.Chat.SENDER,
            ):
                context.bot_data["group_thread"]._queue.put(
                    (_tg_adapter(f), (update, context), {})
                )
            else:
                context.bot_data["pm_thread"]._queue.put(
                    (_tg_adapter(f), (update, context), {})
                )

        tg_handlers.append(handler_type(*args, decorated, **kwargs))
        return decorated

    return decorator


def error_handler(update: tg.Update, context: BoardGameContext):
    try:
        raise context.error
    except Exception as err:
        if type(err) == tg.error.Conflict:
            context.bot.send_message(
                chat_id=os.environ["CREATOR_ID"],
                text="Ошибка: несколько программ подключены к одному боту",
            )
        else:
            tb = "\n".join(
                [
                    traceback.format_exc(limit=10),
                    "",
                    json.dumps(update.to_dict(), indent=2)
                    if update is not None
                    else "No update provided.",
                ]
            )
            tb = tb[max(0, len(tb) - tg.constants.MAX_MESSAGE_LENGTH) :]

            context.bot.send_message(
                chat_id=os.environ["CREATOR_ID"],
                text=tb,
                entities=[tg.MessageEntity(tg.MessageEntity.PRE, 0, len(tb))],
            )


@tg_callback(tg.ext.CommandHandler, "start")
def start(update: tg.Update, context: BoardGameContext):
    if context.args:
        if context.args[0].startswith("pmid"):
            pmsg_id = context.args[0].removeprefix("pmid")
            pm, pm_args, pm_kwargs = context.db.get_pending_message(pmsg_id)
            if pm is not None:
                pm(update, context, *pm_args, **pm_kwargs)
            elif hasattr(core, "on_pm_expired"):
                core.on_pm_expired(update, context)
    else:
        update.effective_chat.send_message(text=context.langtable["main:start-msg"])


@tg_callback(tg.ext.CommandHandler, "settings")
def settings(update: tg.Update, context: BoardGameContext):
    is_anon = context.db.get_anon_mode(update.effective_user)
    update.effective_message.reply_text(
        context.langtable["main:settings-msg"],
        parse_mode=tg.ParseMode.HTML,
        reply_markup=tg.InlineKeyboardMarkup(
            [
                [
                    tg.InlineKeyboardButton(
                        text=context.langtable["main:anonmode-button"].format(
                            state="🟢" if is_anon else "🔴"
                        ),
                        callback_data=core.format_callback_data(
                            "ANON_MODE_OFF" if is_anon else "ANON_MODE_ON",
                            handler_id="MAIN",
                            expected_uid=update.effective_user.id,
                        ),
                    )
                ]
            ]
        ),
    )


@tg_callback(tg.ext.CommandHandler, "stats")
def stats(update: tg.Update, context: BoardGameContext):
    try:
        user_data = context.db[update.effective_user.id]
        update.effective_message.reply_text(
            context.langtable["main:stats"].format(
                name=context.db.get_name(update.effective_user),
                total=user_data["total"],
                wins=user_data["wins"],
                winrate=user_data["wins"] / user_data["total"] if user_data["total"] else 0.0,
            ),
            parse_mode=tg.ParseMode.HTML,
        )
    except BaseException as exc:
        context.dispatcher.dispatch_error(update, exc)


@tg_callback(tg.ext.CommandHandler, core.__name__)
def boardgame_menu(update: tg.Update, context: BoardGameContext) -> None:
    update.effective_chat.send_message(
        context.langtable["main:game-setup"],
        reply_markup=context.menu.encode(update.effective_user),
    )


@tg_callback(tg.ext.InlineQueryHandler)
def send_invite_inline(update: tg.Update, context: BoardGameContext) -> None:
    logging.info(f"Handling inline query: {update.inline_query.query}")

    options = context.menu.tg_decode(update.inline_query.query)
    challenge_desc = context.menu.prettify(options, update.effective_user.language_code)

    match_id = core.create_match_id()

    update.inline_query.answer(
        results=(
            tg.InlineQueryResultPhoto(
                match_id,
                core.INVITE_IMAGE,
                core.INVITE_IMAGE,
                title=context.langtable["main:send-challenge"],
                description=challenge_desc,
                caption=context.langtable["main:invite-msg"].format(
                    name=update.effective_user.name, options=challenge_desc
                ),
                reply_markup=tg.InlineKeyboardMarkup(
                    [
                        [
                            tg.InlineKeyboardButton(
                                text=context.langtable["main:accept-button"],
                                callback_data=core.format_callback_data(
                                    "ACCEPT", [match_id], handler_id="MAIN"
                                ),
                            )
                        ]
                    ]
                ),
            ),
        )
    )


@tg_callback(tg.ext.ChosenInlineResultHandler)
def create_invite(update: tg.Update, context: BoardGameContext):
    options = context.menu.tg_decode(update.chosen_inline_result.query)

    context.db.create_invite(
        update.chosen_inline_result.result_id, update.effective_user, options
    )


@tg_callback(tg.ext.MessageHandler, tg.ext.filters.Filters.regex("^/"))
def unknown(update: tg.Update, context: BoardGameContext):
    if not update.effective_message.text.isascii():
        return

    ratios = []
    d = difflib.SequenceMatcher(a=update.effective_message.text)
    for command in context.langtable["main:cmds"].keys():
        d.set_seq2(command)
        ratios.append((d.ratio(), command))
    suggested = max(ratios, key=lambda x: x[0])[1]
    update.effective_message.reply_text(
        context.langtable["main:unknown-cmd"].format(suggested=suggested)
    )


@tg_callback(tg.ext.CallbackQueryHandler)
def button_callback(update: tg.Update, context: BoardGameContext) -> Optional[tuple]:
    args = core.parse_callback_data(update.callback_query.data)
    logging.debug(f"Handling user input: {args}")
    if (
        args["expected_uid"]
        and args["expected_uid"] != update.callback_query.from_user.id
    ):
        if args["handler_id"] == "MAIN":
            update.callback_query.answer(context.langtable["main:error-popup-msg"])
        else:
            update.callback_query.answer(
                text=context.langtable["main:unexpected-uid"],
                show_alert=True,
            )
        return

    if args["handler_id"] == "MAIN":
        if args["command"] == "NA":
            update.callback_query.answer(text=context.langtable["main:error-popup-msg"])
        elif args["command"] == "ANON_MODE_OFF":
            context.db.set_anon_mode(update.effective_user, False)
            update.callback_query.answer(
                context.langtable["main:anonmode-off"], show_alert=True
            )
            update.effective_message.edit_reply_markup(
                tg.InlineKeyboardMarkup(
                    [
                        [
                            tg.InlineKeyboardButton(
                                text=context.langtable["main:anonmode-button"].format(state="🔴"),
                                callback_data=core.format_callback_data(
                                    "ANON_MODE_ON",
                                    expected_uid=update.effective_user.id,
                                    handler_id="MAIN",
                                ),
                            )
                        ]
                    ]
                )
            )
        elif args["command"] == "ANON_MODE_ON":
            context.db.set_anon_mode(update.effective_user, True)
            update.callback_query.answer(
                context.langtable["main:anonmode-on"], show_alert=True
            )
            update.effective_message.edit_reply_markup(
                tg.InlineKeyboardMarkup(
                    [
                        [
                            tg.InlineKeyboardButton(
                                text=context.langtable["main:anonmode-button"].format(state="🟢"),
                                callback_data=core.format_callback_data(
                                    "ANON_MODE_OFF",
                                    handler_id="MAIN",
                                    expected_uid=update.effective_user.id,
                                ),
                            )
                        ]
                    ]
                )
            )

        elif args["command"] == "PREV":
            options = context.menu.decode(update.effective_message.reply_markup)
            key = args["args"][0]

            values = context.menu[key].available_values(options)
            values_iter = reversed(values)
            for value in values_iter:
                if value == options[key]:
                    try:
                        options[key] = next(values_iter)
                    except StopIteration:
                        options[key] = values[-1]
                    break

            if len(values) > 1:
                notes = []
                if options["mode"] == "online":
                    notes.append(context.langtable["main:queue-len"].format(n=len(context.bot_data["queue"])))
                for value in options.values():
                    notes += [context.langtable[f"{value}:note"]] if f"{value}:note" in context.langtable else []
                notes.extend(("", context.langtable["main:game-setup"]))

                update.effective_message.edit_text(
                    text="\n".join(notes),
                    reply_markup=context.menu.encode(update.effective_user, indexes=options)
                )
            update.callback_query.answer(
                f"{context.langtable['main:chosen-popup']} {context.langtable[key]} - {context.langtable[options[key]]}"
            )

        elif args["command"] == "NEXT":
            options = context.menu.decode(update.effective_message.reply_markup)
            key = args["args"][0]

            values = context.menu[key].available_values(options)
            values_iter = iter(values)
            for value in values_iter:
                if value == options[key]:
                    try:
                        options[key] = next(values_iter)
                    except StopIteration:
                        options[key] = values[0]
                    break

            if len(values) > 1:
                notes = []
                if options["mode"] == "online":
                    notes.append(context.langtable["main:queue-len"].format(n=len(context.bot_data["queue"])))
                for value in options.values():
                    notes += [context.langtable[f"{value}:note"]] if f"{value}:note" in context.langtable else []
                notes.extend(("", context.langtable["main:game-setup"]))

                update.effective_message.edit_text(
                    text="\n".join(notes),
                    reply_markup=context.menu.encode(update.effective_user, indexes=options)
                )
            update.callback_query.answer(
                f"{context.langtable['main:chosen-popup']} {context.langtable[key]} - {context.langtable[options[key]]}"
            )

        elif args["command"] == "DESC":
            key = f"{context.menu.get_value(*args['args'])}:desc"
            print(key)
            update.callback_query.answer(
                context.langtable[key] if key in context.langtable else None,
                show_alert=True
            )

        elif args["command"] == "PLAY":
            options = context.menu.decode(update.effective_message.reply_markup)

            for match in context.bot_data["matches"].values():
                if update.effective_user in match:
                    if isinstance(match, core.PMMatch):
                        msg = (
                            match.msg1
                            if update.effective_user == match.player1
                            else match.msg2
                        )
                    elif isinstance(match, core.GroupMatch):
                        msg = match.msg

                    if (
                        not isinstance(msg, core.InlineMessage)
                        and msg.chat == update.effective_chat
                    ):
                        update.callback_query.answer(
                            context.langtable["main:pending-match-in-same-chat"],
                            show_alert=True,
                        )
                    else:
                        update.callback_query.answer(
                            context.langtable["main:pending-match-in-other-chat"],
                            show_alert=True,
                        )

                    return

            if options["mode"] == "vsbot":
                update.effective_message.edit_text(
                    context.langtable["main:match-found"]
                )
                new_msg = update.effective_chat.send_photo(
                    core.INVITE_IMAGE, caption=context.langtable["main:starting-match"]
                )
                new = core.AIMatch(
                    player1=update.effective_user,
                    player2=context.bot.get_me(),
                    chat1=update.effective_chat.id,
                    msg1=new_msg,
                    options=options,
                    dispatcher=context.dispatcher,
                )
                context.bot_data["matches"][new.id] = new
                try:
                    new.init_turn()
                except BaseException as exc:
                    del context.bot_data["matches"][new.id]
                    if hasattr(core, "ERROR_IMAGE"):
                        new_msg.edit_media(
                            media=tg.InputMediaPhoto(
                                core.ERROR_IMAGE,
                                caption=context.langtable["main:init-error"],
                            )
                        )
                    else:
                        new_msg.edit_caption(
                            caption=context.langtable["main:init-error"]
                        )
                    context.dispatcher.dispatch_error(update, exc)

            elif options["mode"] == "online":
                opponent = get_opponent(context.bot_data["queue"], options)
                logging.info(f"Opponent found: {opponent}")
                if opponent:
                    context.bot_data["queue"].remove(opponent)
                    update.effective_message.edit_text(
                        context.langtable["main:match-found"]
                    )
                    opponent["msg"].edit_text(context.langtable["main:match-found"])
                    new_msg = update.effective_chat.send_photo(
                        core.INVITE_IMAGE,
                        caption=context.langtable["main:starting-match"],
                    )
                    if opponent["chat_id"] == update.effective_chat.id:
                        new = core.GroupMatch(
                            player1=opponent["user"],
                            player2=update.effective_user,
                            chat=opponent["chat_id"],
                            msg=new_msg,
                            options=options,
                            dispatcher=context.dispatcher,
                        )
                        context.bot_data["matches"][new.id] = new
                        try:
                            new.init_turn()
                        except BaseException as exc:
                            del context.bot_data["matches"][new.id]
                            if hasattr(core, "ERROR_IMAGE"):
                                new_msg.edit_media(
                                    media=tg.InputMediaPhoto(
                                        core.ERROR_IMAGE,
                                        caption=context.langtable["main:init-error"],
                                    )
                                )
                            else:
                                new_msg.edit_caption(
                                    caption=context.langtable["main:init-error"]
                                )
                            context.dispatcher.dispatch_error(update, exc)

                    else:
                        opponent_msg = context.bot.send_photo(
                            opponent["chat_id"],
                            core.INVITE_IMAGE,
                            caption=context.langtable["main:starting-match"],
                        )
                        new = core.PMMatch(
                            player1=opponent["user"],
                            player2=update.effective_user,
                            chat1=opponent["chat_id"],
                            chat2=update.effective_chat.id,
                            msg1=new_msg,
                            msg2=opponent_msg,
                            options=options,
                            dispatcher=context.dispatcher,
                        )
                        context.bot_data["matches"][new.id] = new
                        try:
                            new.init_turn()
                        except BaseException as exc:
                            del context.bot_data["matches"][new.id]
                            if hasattr(core, "ERROR_IMAGE"):
                                media = tg.InputMediaPhoto(
                                    core.ERROR_IMAGE,
                                    context.langtable["main:init-error"],
                                )
                                new_msg.edit_media(media=media)
                                opponent_msg.edit_media(media=media)
                            else:
                                new_msg.edit_caption(
                                    caption=context.langtable["main:init-error"]
                                )
                                opponent_msg.edit_caption(
                                    caption=context.langtable["main:init-error"]
                                )

                else:
                    context.bot_data["queue"].append(
                        {
                            "user": update.effective_user,
                            "msg": update.effective_message,
                            "chat_id": update.effective_chat.id,
                            "options": options,
                        }
                    )
                    update.effective_message.edit_text(
                        "\n".join(
                            [
                                context.langtable["main:awaiting-opponent"],
                                "",
                                context.menu.prettify(
                                    options, update.effective_user.language_code
                                ),
                            ]
                        ),
                        reply_markup=tg.InlineKeyboardMarkup(
                            [
                                [
                                    tg.InlineKeyboardButton(
                                        text=context.langtable["main:cancel-button"],
                                        callback_data=core.format_callback_data(
                                            "CANCEL",
                                            handler_id="MAIN",
                                            expected_uid=update.effective_user.id,
                                        ),
                                    )
                                ]
                            ]
                        ),
                    )
            elif options["mode"] == "invite":
                update.effective_message.edit_text(
                    context.langtable["main:invite-sent"]
                )

        elif args["command"] == "CANCEL":
            for index, queued in enumerate(context.bot_data["queue"]):
                if queued["user"] == update.effective_user:
                    del context.bot_data["queue"][index]
                    update.effective_message.edit_text(
                        context.langtable["main:search-cancelled"]
                    )
                    return

            update.callback_query.answer(context.langtable["main:error-popup-msg"])
            update.effective_message.edit_reply_markup()

        elif args["command"] == "ACCEPT":
            challenge = context.db.get_invite(args["args"][0])
            logging.info(f"invite {args['args'][0]}: {challenge}")
            if challenge:
                try:
                    new = core.GroupMatch(
                        player1=challenge["from_user"],
                        player2=update.effective_user,
                        msg=core.InlineMessage(
                            update.callback_query.inline_message_id, context.bot
                        ),
                        options=challenge["options"],
                        dispatcher=context.dispatcher,
                    )
                    context.bot_data["matches"][new.id] = new
                    new.init_turn()
                except BaseException as exc:
                    context.dispatcher.dispatch_error(update, exc)
                    context.bot.edit_message_caption(
                        inline_message_id=update.callback_query.inline_message_id,
                        caption=context.langtable["main:init-error"],
                    )
            else:
                context.bot.edit_message_text(
                    inline_message_id=update.callback_query.inline_message_id,
                    text=context.langtable["main:invite-not-found"],
                )

        else:
            raise ValueError(
                f"unknown command {args['command']} for handler {args['target_id']}"
            )

    elif args["handler_id"] == "core":
        core.KEYBOARD_COMMANDS[args["command"]](update, context, args["args"])
    else:
        if context.bot_data["matches"].get(args["handler_id"]):
            res = context.bot_data["matches"][args["handler_id"]].handle_input(
                args["command"], args["args"]
            )
            update.callback_query.answer(**(res or {}))
        else:
            update.callback_query.answer(
                text=context.langtable["main:game-not-found-error"]
            )


@flask_callback(f"/{os.environ['BOT_TOKEN']}/send-update", methods=["POST"])
def process_update():
    start = time.monotonic_ns()
    dispatcher = flask.current_app.dispatcher
    logging.info(f"Handling update: {flask.request.json}")
    dispatcher.bot_data["pending_updates"][flask.request.json["update_id"]] = start
    dispatcher.process_update(tg.Update.de_json(flask.request.json, dispatcher.bot))
    return "<p>True</p>"


@flask_callback(f"/{os.environ['BOT_TOKEN']}/dynamic/<filename>")
def fetch_dynamic(filename):
    path = os.path.join("images", "temp", filename)
    data = io.BytesIO(open(path, "rb").read())
    os.remove(path)
    return flask.send_file(data, download_name=filename)


@flask_callback(f"/{os.environ['BOT_TOKEN']}/static/<filename>")
def fetch_static(filename):
    return flask.send_file(
        os.path.join("images", "static", filename), download_name=filename
    )


def create_app() -> TelegramBotApp:
    logging.basicConfig(
        format="{levelname} {name}: {message}", style="{", level=logging.INFO
    )

    dispatcher = tg.ext.Dispatcher(
        tg.ext.ExtBot(os.environ["BOT_TOKEN"], defaults=tg.ext.Defaults(quote=True)),
        queue.Queue(),
        context_types=tg.ext.ContextTypes(context=BoardGameContext),
    )
    app = TelegramBotApp(__name__)
    app.dispatcher = dispatcher
    conn = RedisInterface.from_url(os.environ["REDISCLOUD_URL"])
    dispatcher.bot_data.update(
        {
            "matches": {},
            "queue": [],
            "challenges": {},
            "conn": conn,
            "group_thread": tg.ext.DelayQueue(),
            "pm_thread": tg.ext.DelayQueue(burst_limit=20),
            "pending_updates": {},
        }
    )
    core.set_dispatcher(dispatcher)
    if hasattr(core, "init"):
        logging.info("Custom module initializer called")
        core.init(IS_DEBUG)

    for id, match in conn._fetch_matches(core.from_bytes, dispatcher):
        dispatcher.bot_data["matches"][id] = match
    logging.info(f"Fetching matches: {dispatcher.bot_data['matches']}")

    if hasattr(core, "TEXT_COMMANDS"):
        for cmd, f in core.TEXT_COMMANDS.items():
            tg_callback(tg.ext.CommandHandler, cmd)(f)

    for handler in tg_handlers:
        dispatcher.add_handler(handler)
    dispatcher.add_error_handler(error_handler)

    for args, kwargs in flask_callbacks:
        app.add_url_rule(*args, **kwargs)

    for key in core.langtable.keys():
        dispatcher.bot.set_my_commands(
            list(core.langtable[key]["main:cmds"].items()), language_code=key
        )

    dispatcher.bot.set_webhook(
        "/".join([os.environ["HOST_URL"], os.environ["BOT_TOKEN"], "send-update"])
    )
    dispatcher.bot.send_message(chat_id=os.environ["CREATOR_ID"], text="Бот включен")
    logging.info("Application initialized successfully")

    return app


if __name__ == "__main__":
    USAGE_MSG = f"usage: python {__file__} [ run | db ]"
    if len(sys.argv) < 2:
        print(USAGE_MSG)
    elif sys.argv[1] == "db":
        print("Connecting to database...")
        env = json.load(open("debug_env.json"))
        conn: RedisInterface = RedisInterface.from_url(env["REDISCLOUD_URL"])

        os.system("clear")
        print(
            "Board Game Bot CLI\nEnter a command. To get list of commands, type 'help'"
        )

        while True:
            try:
                command, *args = input(">>> ").split(" ")
            except:
                print("\nClosing connection to the database...")
                conn.close()
                break

            if command == "matches":
                count = 0
                memory_sum = 0
                for key in conn.scan_iter(match="match:*"):
                    count += 1
                    match = conn.get(key)
                    memory_sum += len(match)
                    print(
                        "Match",
                        key.split(b":")[1].decode(),
                        f"({len(match)} bytes)",
                        ":",
                    )
                    print(match.decode(errors="replace"), "\n")
                print(
                    f"Total {count} matches ( {round(memory_sum / 1024, 2)} Kbytes) ."
                )

            elif command == "del-match":
                try:
                    conn.delete(f"match:{args[0]}")
                    print(f"Match {args[0]} deleted.")
                except IndexError:
                    conn.delete(*tuple(conn.scan_iter(match="match:*")))
                    print("All matches deleted.")

            elif command == "user-stats":
                print("Number of users per language code:")
                langcodes = {}
                all_users = 0
                for uid in conn.get_user_ids():
                    userdata = conn.get_user_data()
                    langcodes[userdata["lang_code"]] = (
                        langcodes.get(userdata["lang_code"], 0) + 1
                    )
                    all_users += 1
                for lang_code, n in langcodes.items():
                    print(f"  {lang_code}: {n}")
                print(f"\nTotal {all_users} users.")

            elif command == "help":
                print(
                    """
List of commands:
    'matches' - Get information about all ongoing matches.
    'del-match [id]' - Delete a match with a specified ID.
    'user-stats' - Get statistics about users of the bot.
    'help' - Display this message.
                    """
                )
            else:
                print(
                    f"unknown command: {command!r}\nTo get a list of all available commands, type 'help'"
                )
    elif sys.argv[1] == "run":
        os.system("gunicorn 'main:create_app()'")
    else:
        print(USAGE_MSG)
