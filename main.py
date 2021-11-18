import io
import queue
import os, sys
import json
from typing import (
    Optional,
    Callable,
    Type,
    Union,
    cast,
)
import telegram as tg
from telegram.ext import (
    Dispatcher,
    CommandHandler,
    ChatMemberHandler,
    Handler,
    ChosenInlineResultHandler,
    InlineQueryHandler,
    filters,
    ExtBot,
    MessageHandler,
    DelayQueue,
    ContextTypes,
    Defaults,
    CallbackQueryHandler,
)
import chess as core
import difflib
import logging
import re
import traceback
import flask
import time


INLINE_OPTION_PATTERN = re.compile(r"\w+:\d")
flask_callbacks: list[tuple[tuple, dict]] = []
tg_handlers: list[Handler] = []
tg_keyboard_commands: dict[str, core.KeyboardCommand] = {}
core.BoardGameContext.menu = core.MenuFormatter.from_dict(core.OPTIONS)

debug_env_path = os.path.join(os.path.dirname(__file__), "debug_env.json")
IS_DEBUG = os.path.exists(debug_env_path)
if IS_DEBUG:
    with open(debug_env_path) as r:
        os.environ.update(json.load(r))


class TelegramBotApp(flask.Flask):
    dispatcher: Dispatcher


def get_opponent(queue: list[dict], options: dict) -> Optional[dict]:
    for queuepoint in reversed(queue):
        if options == queuepoint["options"]:
            return queuepoint
    return None


def _tg_adapter(f: core.TextCommand) -> core.TextCommand:
    def decorated(update: tg.Update, context: core.BoardGameContext):
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


def flask_callback(*args, **kwargs) -> Callable[[Callable], Callable]:
    def decorator(f: Callable) -> Callable:
        flask_callbacks.append((args, kwargs | {"view_func": f}))
        return f

    return decorator


def tg_callback(
    handler_type: Type[Handler], *args, **kwargs
) -> Callable[[core.TextCommand], core.TextCommand]:
    def decorator(f: core.TextCommand) -> core.TextCommand:
        def decorated(update: tg.Update, context: core.BoardGameContext):

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

        tg_handlers.append(handler_type(*args, decorated, **kwargs))  # type: ignore
        return f

    return decorator


def keyboard_command(f: core.KeyboardCommand) -> core.KeyboardCommand:
    tg_keyboard_commands[f.__name__.upper()] = f
    return f


def error_handler(update: object, context: core.BoardGameContext):
    try:
        raise cast(BaseException, context.error)
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
                    if isinstance(update, tg.Update)
                    else "No update provided.",
                ]
            )
            tb = tb[: min(len(tb), tg.constants.MAX_MESSAGE_LENGTH)]

            context.bot.send_message(
                chat_id=os.environ["CREATOR_ID"],
                text=tb,
                entities=[tg.MessageEntity(tg.MessageEntity.PRE, 0, len(tb))],
            )


@tg_callback(ChatMemberHandler)
@tg_callback(CommandHandler, "start")
def start(update: tg.Update, context: core.BoardGameContext):
    if context.args:
        if context.args[0].startswith("pmid"):
            pmsg_id = context.args[0].removeprefix("pmid")
            pm = context.db.get_pending_message(pmsg_id)
            if pm is not None:
                pm[0](update, context, *pm[1])
            elif hasattr(core, "on_pm_expired"):
                core.on_pm_expired(update, context)  # type: ignore
    else:
        try:
            cast(tg.Chat, update.effective_chat).send_message(
                text=context.langtable["main:start-msg"]
            )
        except tg.error.Unauthorized:
            pass


@tg_callback(CommandHandler, "settings")
def settings(update: tg.Update, context: core.BoardGameContext):
    assert update.effective_user and update.effective_message
    is_anon = context.db.get_user_data(update.effective_user.id)["is_anon"]
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
                        callback_data=str(
                            core.CallbackData(
                                "ANON_MODE_OFF" if is_anon else "ANON_MODE_ON",
                                handler_id="MAIN",
                                expected_uid=update.effective_user.id,
                            )
                        ),
                    )
                ]
            ]
        ),
    )


@tg_callback(CommandHandler, "stats")
def stats(update: tg.Update, context: core.BoardGameContext):
    assert update.effective_user and update.effective_message
    user_data = context.db.get_user_data(update.effective_user.id)
    update.effective_message.reply_text(
        context.langtable["main:stats"].format(
            name=context.db.get_name(update.effective_user),
            total=user_data["total"],
            wins=user_data["wins"],
            winrate=user_data["wins"] * 100 / user_data["total"]
            if user_data["total"]
            else 0.0,
        ),
        parse_mode=tg.ParseMode.HTML,
    )


@tg_callback(CommandHandler, "clear")
def clear_match_cache(update: tg.Update, context: core.BoardGameContext):
    assert update.effective_message and update.effective_user
    if update.effective_user.id == int(os.environ["CREATOR_ID"]):
        n = 0
        if context.args:
            for match_id in context.args:
                n += context.db.delete(f"match:{match_id}")
                del context.bot_data["matches"][match_id]
        else:
            context.bot_data["matches"].clear()
            for match_id in context.db.scan_iter(match="match:*"):
                n += context.db.delete(match_id)

        update.effective_message.reply_text(f"{n} games deleted")


@tg_callback(CommandHandler, core.__name__)
def boardgame_menu(update: tg.Update, context: core.BoardGameContext) -> None:
    assert update.effective_chat and update.effective_user
    update.effective_chat.send_message(
        context.menu.format_notes(context),
        reply_markup=context.menu.encode(update.effective_user),
    )


@tg_callback(InlineQueryHandler)
def send_invite_inline(update: tg.Update, context: core.BoardGameContext) -> None:
    assert update.inline_query and update.effective_user
    logging.info(f"Handling inline query: {update.inline_query.query}")

    options = context.menu.tg_decode(update.inline_query.query)
    challenge_desc = context.menu.prettify(
        options, cast(str, update.effective_user.language_code)
    )

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
                                callback_data=str(
                                    core.CallbackData(
                                        "ACCEPT", args=[match_id], handler_id="MAIN"
                                    )
                                ),
                            )
                        ]
                    ]
                ),
            ),
        )
    )


@tg_callback(ChosenInlineResultHandler)
def create_invite(update: tg.Update, context: core.BoardGameContext):
    assert update.chosen_inline_result and update.effective_user
    options = context.menu.tg_decode(update.chosen_inline_result.query)

    context.db.create_invite(
        update.chosen_inline_result.result_id, update.effective_user, options
    )


@tg_callback(MessageHandler, filters.Filters.regex("^/"))
def unknown(update: tg.Update, context: core.BoardGameContext):
    assert update.effective_message and update.effective_message.text
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


@keyboard_command
def anon_mode_off(update: tg.Update, context: core.BoardGameContext, args: list[str]):
    assert update.effective_user and update.effective_message
    context.db.set_anon_mode(update.effective_user, False)
    cast(tg.CallbackQuery, update.callback_query).answer(
        context.langtable["main:anonmode-off"], show_alert=True
    )
    update.effective_message.edit_reply_markup(
        tg.InlineKeyboardMarkup(
            [
                [
                    tg.InlineKeyboardButton(
                        text=context.langtable["main:anonmode-button"].format(
                            state="🔴"
                        ),
                        callback_data=str(
                            core.CallbackData(
                                "ANON_MODE_ON",
                                expected_uid=update.effective_user.id,
                                handler_id="MAIN",
                            )
                        ),
                    )
                ]
            ]
        )
    )


@keyboard_command
def anon_mode_on(update: tg.Update, context: core.BoardGameContext, args: list[str]):
    assert update.effective_message and update.effective_user
    context.db.set_anon_mode(update.effective_user, True)
    cast(tg.CallbackQuery, update.callback_query).answer(
        context.langtable["main:anonmode-on"], show_alert=True
    )
    update.effective_message.edit_reply_markup(
        tg.InlineKeyboardMarkup(
            [
                [
                    tg.InlineKeyboardButton(
                        text=context.langtable["main:anonmode-button"].format(
                            state="🟢"
                        ),
                        callback_data=str(
                            core.CallbackData(
                                "ANON_MODE_OFF",
                                handler_id="MAIN",
                                expected_uid=update.effective_user.id,
                            )
                        ),
                    )
                ]
            ]
        )
    )


@keyboard_command
def prev(update: tg.Update, context: core.BoardGameContext, args: list[str]):
    assert update.effective_message and update.effective_user and update.callback_query
    options = cast(dict[str, Optional[str]], context.menu.decode(
        cast(tg.InlineKeyboardMarkup, update.effective_message.reply_markup)
    ))
    key = args[0]

    values = context.menu[key].available_values(options)
    values_iter = reversed(values)
    for value in values_iter:
        if value == options[key]:
            try:
                options[key] = __builtins__["next"](values_iter)    # type: ignore
            except StopIteration:
                options[key] = values[-1]
            break

    if len(values) > 1:
        update.effective_message.edit_text(
            text=context.menu.format_notes(context, options=options),
            reply_markup=context.menu.encode(update.effective_user, indexes=options),
        )
    update.callback_query.answer(
        f"{context.langtable['main:chosen-popup']} {context.langtable[key]} - {context.langtable[options[key]]}"
    )


@keyboard_command
def next(update: tg.Update, context: core.BoardGameContext, args: list[str]):
    assert update.effective_message and update.effective_user and update.callback_query
    options = cast(dict[str, Optional[str]], context.menu.decode(
        cast(tg.InlineKeyboardMarkup, update.effective_message.reply_markup)
    ))
    key = args[0]

    values = context.menu[key].available_values(options)
    values_iter = iter(values)
    for value in values_iter:
        if value == options[key]:
            try:
                options[key] = __builtins__["next"](values_iter)   # type: ignore
            except StopIteration:
                options[key] = values[0]
            break

    if len(values) > 1:
        update.effective_message.edit_text(
            text=context.menu.format_notes(context, options=options),
            reply_markup=context.menu.encode(update.effective_user, indexes=options),
        )
    update.callback_query.answer(
        f"{context.langtable['main:chosen-popup']} {context.langtable[key]} - {context.langtable[options[key]]}"
    )


@keyboard_command
def desc(update: tg.Update, context: core.BoardGameContext, args: list[str]):
    key = f"{context.menu.get_value(args[0], int(args[1]))}:desc"
    cast(tg.CallbackQuery, update.callback_query).answer(
        context.langtable[key] if key in context.langtable else None,
        show_alert=True,
    )


@keyboard_command
def restoremenu(update: tg.Update, context: core.BoardGameContext, args: list[str]):
    cast(tg.Message, update.effective_message).delete()
    boardgame_menu(update, context)


@keyboard_command
def play(update: tg.Update, context: core.BoardGameContext, args: list[str]):
    assert (
        isinstance(context.user_data, dict)
        and update.effective_message
        and update.callback_query
        and update.effective_user
        and update.effective_chat
    )
    if args:
        options = context.user_data["opponent"]["options"]
    elif context.menu.is_valid(update.effective_message.reply_markup):
        options = context.menu.decode(
            cast(tg.InlineKeyboardMarkup, update.effective_message.reply_markup)
        )
    else:
        restoremenu(update, context, args)

    for match in context.bot_data["matches"].values():
        if update.effective_user in match:
            if isinstance(match, core.PMMatch):
                msg = cast(
                    Union[core.InlineMessage, tg.Message],
                    match.msg1
                    if update.effective_user == match.player1
                    else match.msg2,
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
        update.effective_message.edit_text(context.langtable["main:match-found"])
        new_msg = update.effective_chat.send_photo(
            core.INVITE_IMAGE, caption=context.langtable["main:starting-match"]
        )
        new: core.BaseMatch = core.AIMatch(
            update.effective_user,
            context.bot.get_me(),
            new_msg,
            None,
            options=options,
            dispatcher=context.dispatcher,
        )
        context.bot_data["matches"][new.id] = new
        try:
            new.init_turn()
        except Exception as exc:
            del context.bot_data["matches"][new.id]
            if hasattr(core, "ERROR_IMAGE"):
                new_msg.edit_media(
                    media=tg.InputMediaPhoto(
                        core.ERROR_IMAGE,   # type: ignore
                        caption=context.langtable["main:init-error"],
                    )
                )
            else:
                new_msg.edit_caption(caption=context.langtable["main:init-error"])
            context.dispatcher.dispatch_error(update, exc)

    elif options["mode"] == "online":
        if args:
            opponent = context.user_data["opponent"]
            del context.user_data["opponent"]
        else:
            opponent = get_opponent(context.bot_data["queue"], options)
        logging.info(f"Opponent found: {opponent}")

        if opponent:
            if opponent["user"] == update.effective_user and not args:
                context.user_data["opponent"] = opponent
                update.effective_message.edit_text(
                    text=context.langtable["main:same-player-warning"],
                    reply_markup=tg.InlineKeyboardMarkup(
                        [
                            [
                                tg.InlineKeyboardButton(
                                    text=context.langtable["main:continue-button"],
                                    callback_data=str(core.CallbackData(
                                        "PLAY",
                                        args=["1"],
                                        handler_id="MAIN",
                                        expected_uid=update.effective_user.id,
                                    )),
                                ),
                                tg.InlineKeyboardButton(
                                    text=context.langtable["main:cancel-button"],
                                    callback_data=str(core.CallbackData(
                                        "RESTOREMENU",
                                        handler_id="MAIN",
                                        expected_uid=update.effective_user.id,
                                    )),
                                ),
                            ]
                        ]
                    ),
                )
            else:
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
                        opponent["user"],
                        update.effective_user,
                        new_msg,
                        options=options,
                        dispatcher=context.dispatcher,
                    )
                    context.bot_data["matches"][new.id] = new
                    try:
                        new.init_turn()
                    except Exception as exc:
                        del context.bot_data["matches"][new.id]
                        if hasattr(core, "ERROR_IMAGE"):
                            new_msg.edit_media(
                                media=tg.InputMediaPhoto(
                                    core.ERROR_IMAGE,   # type: ignore
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
                        update.effective_user,
                        opponent["user"],
                        new_msg,
                        opponent_msg,
                        options=options,
                        dispatcher=context.dispatcher,
                    )
                    context.bot_data["matches"][new.id] = new
                    try:
                        new.init_turn()
                    except Exception as exc:
                        del context.bot_data["matches"][new.id]
                        if hasattr(core, "ERROR_IMAGE"):
                            media = tg.InputMediaPhoto(
                                core.ERROR_IMAGE,    # type: ignore
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
                            options, cast(str, update.effective_user.language_code)
                        ),
                    ]
                ),
                reply_markup=tg.InlineKeyboardMarkup(
                    [
                        [
                            tg.InlineKeyboardButton(
                                text=context.langtable["main:cancel-button"],
                                callback_data=str(core.CallbackData(
                                    "CANCEL",
                                    handler_id="MAIN",
                                    expected_uid=update.effective_user.id,
                                )),
                            )
                        ]
                    ]
                ),
            )
    elif options["mode"] == "invite":
        update.effective_message.edit_text(context.langtable["main:invite-sent"])


@keyboard_command
def cancel(update: tg.Update, context: core.BoardGameContext, args: list[str]):
    assert update.effective_message and update.callback_query and update.effective_user
    for index, queued in enumerate(context.bot_data["queue"]):
        if queued["user"] == update.effective_user:
            del context.bot_data["queue"][index]
            update.effective_message.edit_text(
                context.langtable["main:search-cancelled"]
            )
            return

    update.callback_query.answer(context.langtable["main:error-popup-msg"])
    update.effective_message.edit_reply_markup()


@keyboard_command
def remove_menu(update: tg.Update, context: core.BoardGameContext, args: list[str]):
    cast(tg.Message, update.effective_message).edit_text(context.langtable["main:search-cancelled"])


@keyboard_command
def accept(update: tg.Update, context: core.BoardGameContext, args: list[str]):
    assert update.effective_user and update.callback_query
    challenge = context.db.get_invite(args[0])
    logging.info(f"invite {args[0]}: {challenge}")
    if challenge:
        try:
            new = core.GroupMatch(
                challenge["from_user"],
                update.effective_user,
                core.InlineMessage(
                    cast(str, update.callback_query.inline_message_id), context.bot
                ),
                options=challenge["options"],
                dispatcher=context.dispatcher,
            )
            context.bot_data["matches"][new.id] = new
            new.init_turn()
        except Exception as exc:
            context.dispatcher.dispatch_error(update, exc)
            context.bot.edit_message_caption(
                inline_message_id=cast(str, update.callback_query.inline_message_id),
                caption=context.langtable["main:init-error"],
            )
    else:
        context.bot.edit_message_text(
            inline_message_id=update.callback_query.inline_message_id,
            text=context.langtable["main:invite-not-found"],
        )


@tg_callback(CallbackQueryHandler)
def button_callback(update: tg.Update, context: core.BoardGameContext):
    assert update.callback_query
    args = core.CallbackData.decode(cast(str, update.callback_query.data))
    logging.debug(f"Handling user input: {args}")
    if (
        args.expected_uid
        and args.expected_uid != update.callback_query.from_user.id
    ):
        update.callback_query.answer(
            text=context.langtable["main:unexpected-uid"],
            show_alert=True,
        )
        return

    if args.handler_id == "MAIN":
        tg_keyboard_commands[args.command](update, context, args.args)
    else:
        if context.bot_data["matches"].get(args.handler_id):
            res = context.bot_data["matches"][args.handler_id].handle_input(
                args.command, args.args
            )
            update.callback_query.answer(*(res or ()))
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

    dispatcher = Dispatcher(
        ExtBot(os.environ["BOT_TOKEN"], defaults=Defaults(quote=True)),
        queue.Queue(),
        context_types=ContextTypes(context=core.BoardGameContext),
    )
    app = TelegramBotApp(__name__)
    app.dispatcher = dispatcher
    conn = cast(core.RedisInterface, core.RedisInterface.from_url(
        os.environ["REDISCLOUD_URL"]
    ))
    dispatcher.bot_data.update(
        {
            "matches": {},
            "queue": [],
            "challenges": {},
            "conn": conn,
            "group_thread": DelayQueue(),
            "pm_thread": DelayQueue(burst_limit=20),
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
            tg_handlers.insert(0, CommandHandler(cmd, f))

    if hasattr(core, "KEYBOARD_COMMANDS"):
        tg_keyboard_commands.update(core.KEYBOARD_COMMANDS)

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
        conn = cast(core.RedisInterface, core.RedisInterface.from_url(env["REDISCLOUD_URL"]))

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
                    match = cast(bytes, conn.get(key))
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
                langcodes: dict[str, int] = {}
                all_users = 0
                for uid in conn.get_user_ids():
                    userdata = conn[uid]
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
