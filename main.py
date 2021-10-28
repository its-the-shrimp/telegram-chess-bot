import io
import queue
import os
import json
from typing import Optional
import telegram as tg
import chess as core
import bot_utils
import difflib
import logging
import re
import traceback
import flask
import time

debug_env_path = os.path.join(os.path.dirname(__file__), "debug_env.json")
INLINE_OPTION_PATTERN = re.compile(r"\w+:\d")
IS_DEBUG = os.path.exists(debug_env_path)

bot_utils.BoardGameContext.menu = bot_utils.MenuFormatter.from_dict(core.OPTIONS)
logging.basicConfig(
    format="{levelname} {name}: {message}", style="{", level=logging.DEBUG
)

try:
    os.mkdir(os.path.join(os.path.dirname(__file__), "images", "temp"))
except FileExistsError:
    pass


class TelegramBotApp(flask.Flask):
    dispatcher: tg.ext.Dispatcher


def get_opponent(queue: list[dict], options: dict) -> Optional[dict]:
    for queuepoint in reversed(queue):
        if options == queuepoint["options"]:
            return queuepoint


def _tg_adapter(f: core.TelegramCallback) -> core.TelegramCallback:
    def decorated(update: tg.Update, context: bot_utils.BoardGameContext):
        try:
            return f(update, context)
        except Exception as exc:
            context.dispatcher.dispatch_error(update, exc)

    return decorated


def tg_adapter(f: core.TelegramCallback) -> core.TelegramCallback:
    def decorated(update: tg.Update, context: bot_utils.BoardGameContext):

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

    return decorated


def error_handler(update: tg.Update, context: bot_utils.BoardGameContext):
    try:
        raise context.error
    except Exception as err:
        if type(err) == tg.error.Conflict:
            context.bot.send_message(
                chat_id=os.environ["CREATOR_ID"],
                text="Ошибка: несколько программ подключены к одному боту",
            )
        else:
            tb = (
                traceback.format_exc(limit=10)
                + "\n\n"
                + json.dumps(update.to_dict(), indent=2)
            )
            tb = tb[max(0, len(tb) - tg.constants.MAX_MESSAGE_LENGTH) :]

            context.bot.send_message(
                chat_id=os.environ["CREATOR_ID"],
                text=tb,
                entities=[tg.MessageEntity(tg.MessageEntity.PRE, 0, len(tb))],
            )

        if IS_DEBUG and not isinstance(err, tg.error.Unauthorized):
            raise err


@tg_adapter
def start(update: tg.Update, context: bot_utils.BoardGameContext):
    if context.args:
        if context.args[0].startswith("pmid"):
            pmsg_id = context.args[0].removeprefix("pmid")
            pm = context.db.get_pending_message(pmsg_id)
            if pm is not None:
                pm[0](update, context, *pm[1])
            elif "pm_expire_hook" in context.bot_data:
                context.bot_data["pm_expire_hook"](update, context)
    else:
        update.effective_chat.send_message(text=context.langtable["start-msg"])


@tg_adapter
def unknown(update: tg.Update, context: bot_utils.BoardGameContext):
    if not update.effective_message.text.isascii():
        return

    ratios = []
    d = difflib.SequenceMatcher(a=update.effective_message.text)
    for command in context.langtable["cmds"].keys():
        d.set_seq2(command)
        ratios.append((d.ratio(), command))
    suggested = max(ratios, key=lambda x: x[0])[1]
    update.effective_message.reply_text(
        context.langtable["unknown-cmd"].format(suggested=suggested)
    )


@tg_adapter
def settings(update: tg.Update, context: bot_utils.BoardGameContext):
    is_anon = context.db.get_anon_mode(update.effective_user)
    update.effective_message.reply_text(
        context.langtable["settings-msg"],
        parse_mode=tg.ParseMode.HTML,
        reply_markup=tg.InlineKeyboardMarkup(
            [
                [
                    tg.InlineKeyboardButton(
                        text=context.langtable["anonmode-button"].format(
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


@tg_adapter
def boardgame_menu(update: tg.Update, context: bot_utils.BoardGameContext) -> None:
    update.effective_chat.send_message(
        context.langtable["game-setup"],
        reply_markup=context.menu.encode(update.effective_user),
    )


@tg_adapter
def send_invite_inline(update: tg.Update, context: bot_utils.BoardGameContext) -> None:
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
                title=context.langtable["send-challenge"],
                description=challenge_desc,
                caption=context.langtable["invite-msg"].format(
                    name=update.effective_user.name, options=challenge_desc
                ),
                reply_markup=tg.InlineKeyboardMarkup(
                    [
                        [
                            tg.InlineKeyboardButton(
                                text=context.langtable["accept-button"],
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


@tg_adapter
def create_invite(update: tg.Update, context: bot_utils.BoardGameContext):
    options = context.menu.tg_decode(update.chosen_inline_result.query)

    context.db.create_invite(
        update.chosen_inline_result.result_id, update.effective_user, options
    )


@tg_adapter
def button_callback(
    update: tg.Update, context: bot_utils.BoardGameContext
) -> Optional[tuple]:
    args = core.parse_callback_data(update.callback_query.data)
    logging.debug(f"Handling user input: {args}")
    if (
        args["expected_uid"]
        and args["expected_uid"] != update.callback_query.from_user.id
    ):
        if args["target_id"] == "MAIN":
            update.callback_query.answer(context.langtable["error-popup-msg"])
        else:
            update.callback_query.answer(
                text=context.langtable["unexpected-uid"],
                show_alert=True,
            )
        return

    if args["target_id"] == "MAIN":
        if args["command"] == "NA":
            update.callback_query.answer(text=context.langtable["error-popup-msg"])
        elif args["command"] == "ANON_MODE_OFF":
            context.db.set_anon_mode(update.effective_user, False)
            update.callback_query.answer(
                context.langtable["anonmode-off"], show_alert=True
            )
            update.effective_message.edit_reply_markup(
                tg.InlineKeyboardMarkup(
                    [
                        [
                            tg.InlineKeyboardButton(
                                text=context.langtable["anonmode-button"].format(
                                    state="🟢"
                                    if context.db.get_anon_mode(update.effective_user)
                                    else "🔴"
                                ),
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
                context.langtable["anonmode-on"], show_alert=True
            )
            update.effective_message.edit_reply_markup(
                tg.InlineKeyboardMarkup(
                    [
                        [
                            tg.InlineKeyboardButton(
                                text=context.langtable["anonmode-button"].format(
                                    state="🟢"
                                    if context.db.get_anon_mode(update.effective_user)
                                    else "🔴"
                                ),
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
                update.effective_message.edit_reply_markup(
                    context.menu.encode(update.effective_user, indexes=options)
                )
            update.callback_query.answer(
                f"{context.langtable['chosen-popup']} {context.langtable[key]} - {context.langtable[options[key]]}"
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
                update.effective_message.edit_reply_markup(
                    context.menu.encode(update.effective_user, indexes=options)
                )
            update.callback_query.answer(
                f"{context.langtable['chosen-popup']} {context.langtable[key]} - {context.langtable[options[key]]}"
            )

        elif args["command"] == "DESC":
            update.callback_query.answer()

        elif args["command"] == "PLAY":
            options = context.menu.decode(update.effective_message.reply_markup)

            if options["mode"] == "vsbot":
                update.effective_message.edit_text(context.langtable["match-found"])
                new = core.AIMatch(
                    player1=update.effective_user,
                    player2=context.bot.get_me(),
                    chat1=update.effective_chat.id,
                    options=options,
                    dispatcher=context.dispatcher,
                )
                context.bot_data["matches"][new.id] = new
                new.init_turn()
            elif options["mode"] == "online":
                opponent = get_opponent(context.bot_data["queue"], options)
                logging.info(f"Opponent found: {opponent}")
                if opponent:
                    context.bot_data["queue"].remove(opponent)
                    update.effective_message.edit_text(context.langtable["match-found"])
                    opponent["msg"].edit_text(context.langtable["match-found"])

                    if opponent["chat_id"] == update.effective_chat.id:
                        new = core.GroupMatch(
                            player1=opponent["user"],
                            player2=update.effective_user,
                            chat=opponent["chat_id"],
                            options=options,
                            dispatcher=context.dispatcher,
                        )
                    else:
                        new = core.PMMatch(
                            player1=opponent["user"],
                            player2=update.effective_user,
                            chat1=opponent["chat_id"],
                            chat2=update.effective_chat.id,
                            options=options,
                            dispatcher=context.dispatcher,
                        )
                    context.bot_data["matches"][new.id] = new
                    new.init_turn()
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
                                context.langtable["awaiting-opponent"],
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
                                        text=context.langtable["cancel-button"],
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
                update.effective_message.edit_text(context.langtable["invite-sent"])

        elif args["command"] == "CANCEL":
            for index, queued in enumerate(context.bot_data["queue"]):
                if queued["user"] == update.effective_user:
                    del context.bot_data["queue"][index]
                    update.effective_message.edit_text(
                        context.langtable["search-cancelled"]
                    )
                    return

            update.callback_query.answer(context.langtable["error-popup-msg"])
            update.effective_message.edit_reply_markup()

        elif args["command"] == "ACCEPT":
            challenge = context.db.get_invite(args["args"][0])
            logging.info(f"invite {args['args'][0]}: {challenge}")
            if challenge:
                new = core.GroupMatch(
                    challenge["from_user"],
                    update.effective_user,
                    None,
                    msg=core.InlineMessageAdapter(
                        update.callback_query.inline_message_id, context.bot
                    ),
                    options=challenge["options"],
                    dispatcher=context.dispatcher,
                )
                context.bot_data["matches"][new.id] = new
                new.init_turn()
            else:
                context.bot.edit_message_text(
                    inline_message_id=update.callback_query.inline_message_id,
                    text=context.langtable["error-msg"],
                )

        else:
            raise ValueError(
                f"unknown command {args['command']} for handler {args['target_id']}"
            )

    elif args["target_id"] == "core":
        core.KEYBOARD_BUTTONS[args["command"]](update, context, args["args"])
    else:
        if context.bot_data["matches"].get(args["target_id"]):
            res = context.bot_data["matches"][args["target_id"]].handle_input(
                args["command"], args["args"]
            )
            res = res if res else (None, False)
            if context.bot_data["matches"][args["target_id"]].result != "*":
                del context.bot_data["matches"][args["target_id"]]
            update.callback_query.answer(text=res[0], show_alert=res[1])
        else:
            update.callback_query.answer(text=context.langtable["game-not-found-error"])


def create_app() -> flask.Flask:
    dispatcher = tg.ext.Dispatcher(
        tg.Bot(os.environ["BOT_TOKEN"], defaults=tg.ext.Defaults(quote=True)),
        queue.Queue(),
        context_types=tg.ext.ContextTypes(context=bot_utils.BoardGameContext),
    )
    conn = bot_utils.RedisInterface.from_url(os.environ["REDISCLOUD_URL"])
    core.init(IS_DEBUG, conn)

    dispatcher.bot_data.update(
        {
            "matches": {},
            "queue": [],
            "challenges": {},
            "conn": conn,
            "group_thread": tg.ext.DelayQueue(),
            "pm_thread": tg.ext.DelayQueue(burst_limit=20),
        }
    )
    for id, match in conn._fetch_matches(core.from_bytes, dispatcher):
        dispatcher.bot_data["matches"][id] = match

    dispatcher.add_handler(tg.ext.InlineQueryHandler(send_invite_inline))
    dispatcher.add_handler(tg.ext.ChosenInlineResultHandler(create_invite))
    dispatcher.add_handler(tg.ext.CallbackQueryHandler(button_callback))
    dispatcher.add_handler(tg.ext.CommandHandler("start", start))
    dispatcher.add_handler(tg.ext.CommandHandler(core.__name__, boardgame_menu))
    dispatcher.add_handler(tg.ext.CommandHandler("settings", settings))
    dispatcher.add_handler(
        tg.ext.MessageHandler(tg.ext.filters.Filters.regex("^/"), unknown)
    )
    dispatcher.add_error_handler(error_handler)

    for key in core.langtable.keys():
        dispatcher.bot.set_my_commands(
            list(core.langtable[key]["cmds"].items()), language_code=key
        )

    app = flask.Flask(__name__)

    def process_update():
        logging.info(f"Handling update: {flask.request.json}")
        start = time.monotonic_ns()
        dispatcher.process_update(tg.Update.de_json(flask.request.json, dispatcher.bot))
        end = time.monotonic_ns()
        logging.info(
            f"Handled update #{flask.request.json['update_id']} in {(end - start) // 1000000} ms"
        )
        return "<p>True</p>"

    def fetch_dynamic(filename):
        path = os.path.join("images", "temp", filename)
        data = io.BytesIO(open(path, "rb").read())
        os.remove(path)
        return flask.send_file(data, download_name=filename)

    def fetch_static(filename):
        return flask.send_file(
            os.path.join("images", "static", filename), download_name=filename
        )

    app.add_url_rule(
        f"/{os.environ['BOT_TOKEN']}/dynamic/<filename>", view_func=fetch_dynamic
    )
    app.add_url_rule(
        f"/{os.environ['BOT_TOKEN']}/static/<filename>", view_func=fetch_static
    )
    app.add_url_rule(
        f"/{os.environ['BOT_TOKEN']}/send-update",
        view_func=process_update,
        methods=["POST"],
    )
    dispatcher.bot.set_webhook(
        "/".join([os.environ["HOST_URL"], os.environ["BOT_TOKEN"], "send-update"])
    )
    dispatcher.bot.send_message(chat_id=os.environ["CREATOR_ID"], text="Бот включен")

    app.dispatcher = dispatcher
    return app


if __name__ == "__main__":
    os.system("gunicorn 'main:create_app()'")
