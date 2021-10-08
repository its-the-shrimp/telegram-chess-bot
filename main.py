import io
import time
import os
import json
from typing import Callable, Optional
import telegram as tg
import chess as module
import db_utils
import difflib
import gzip
import logging
import re
import traceback
import flask

debug_env_path = os.path.join(os.path.dirname(__file__), "debug_env.json")
INLINE_OPTION_PATTERN = re.compile(r"\w+:\d")
IS_DEBUG = os.path.exists(debug_env_path)

if IS_DEBUG:
    loglevel = logging.DEBUG
    with open(debug_env_path) as r:
        os.environ.update(json.load(r))
else:
    loglevel = logging.INFO


class ExtFormatter(logging.Formatter):
    last_row_created: int = time.monotonic_ns()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        cur_row_created = time.monotonic_ns()
        record.interval = (cur_row_created - self.last_row_created) // 10**6
        self.last_row_created = cur_row_created
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(ExtFormatter("[{interval} ms] {levelname} {name}:\n\t{message}\n", style="{"))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

try:
    os.mkdir(os.path.join(os.path.dirname(__file__), "images", "temp"))
except FileExistsError:
    pass


group_thread = tg.ext.DelayQueue()
pm_thread = tg.ext.DelayQueue()


def stop_bot(_, frame):
        updater = frame.f_locals["self"]
        counter = {}
        for key in updater.persistence.conn.keys(pattern="*:lang"):
            key = updater.persistence.conn.get(key).decode()
            if key:
                counter[key] += 1
            else:
                counter[key] = 1

        updater.bot.send_message(
            text="\n".join([f"{k}: {v}" for k, v in counter.items()]),
            chat_id=os.environ["CREATOR_ID"],
        )
        group_thread.stop()
        pm_thread.stop()
        exit()


def get_opponent(queue: list[dict], options: dict) -> Optional[dict]:
    for queuepoint in reversed(queue):
        if options == queuepoint["options"]:
            return queuepoint


def avoid_spam(f: Callable) -> Callable:
    def decorated(update: tg.Update, context: db_utils.RedisContext):

        if getattr(update.effective_chat, "type", tg.Chat.PRIVATE) in (
            tg.Chat.PRIVATE,
            tg.Chat.SENDER,
        ):
            pm_thread._queue.put((tg_adapter(f), (update, context), {}))
        else:
            group_thread._queue.put((tg_adapter(f), (update, context), {}))

    return decorated


def tg_adapter(
    f: Callable[[tg.Update, db_utils.RedisContext], None]
) -> Callable[[tg.Update, db_utils.RedisContext], None]:
    def decorated(update: tg.Update, context: db_utils.RedisContext):
        context.db.cache_user_data(update.effective_user)
        context.langtable = module.langtable[update.effective_user.language_code]

        return f(update, context)

    return decorated


def error_handler(update: tg.Update, context: db_utils.RedisContext):
    try:
        raise context.error
    except Exception as err:
        if type(err) == tg.error.Conflict:
            context.bot.send_message(
                chat_id=os.environ["CREATOR_ID"],
                text="Ошибка: несколько программ подключены к одному боту",
            )
        else:
            tb = traceback.format_exc(limit=10) + "\n\n\n" + str(update.to_dict())
            tb = tb[max(0, len(tb) - tg.constants.MAX_MESSAGE_LENGTH) :]

            context.bot.send_message(chat_id=os.environ["CREATOR_ID"], text=tb)

        if IS_DEBUG:
            raise err


def decode_data(_, obj: dict) -> dict:
    return {k: gzip.decompress(v).decode() for k, v in obj.items()}


def parse_callbackquery_data(data: str) -> dict:
    data = data.split("\n")
    res = {
        "expected_uid": int(data[0]) if data[0] else None,
        "target_id": data[1],
        "args": [],
    }
    for argument in data[2].split("#"):
        if argument.isdigit():
            res["args"].append(int(argument))
        elif argument:
            res["args"].append(argument)
        else:
            res["args"].append(None)

    return res


def prettify_options(options: dict[str, int], lang_code: str) -> str:
    langtable = module.langtable[lang_code]
    res = langtable["options-title"] + "\n"
    for k, v in options.items():
        options_list = tuple(module.OPTIONS[k]["options"].keys())
        res += langtable[options_list[v]] + ", "

    return res.removeprefix(", ")


def format_options(
    user: tg.User, indexes: dict[str, int] = None
) -> tg.InlineKeyboardMarkup:
    res = []
    if indexes is None:
        indexes = {name: 0 for name in module.OPTIONS}

    for name, option in module.OPTIONS.items():
        chosen = tuple(option["options"].keys())[indexes.get(name, 0)]
        conditions_passed = True
        for c_option, c_func in option["conditions"].items():
            if not c_func(
                tuple(module.OPTIONS[c_option]["options"].keys())[indexes[c_option]]
            ):
                conditions_passed = False

        if conditions_passed:
            res.append(
                [
                    tg.InlineKeyboardButton(
                        text="◀️", callback_data=f"{user.id}\nMAIN\nPREV#{name}"
                    ),
                    tg.InlineKeyboardButton(
                        text=module.langtable[user.language_code][chosen],
                        callback_data=f"{user.id}\nMAIN\nDESC#{name}#{indexes.get(name, 0)}",
                    ),
                    tg.InlineKeyboardButton(
                        text="▶️", callback_data=f"{user.id}\nMAIN\nNEXT#{name}"
                    ),
                ]
            )

    if indexes["mode"] == tuple(module.OPTIONS["mode"]["options"]).index("invite"):
        inline_query = " ".join([f"{k}:{v}" for k, v in indexes.items() if k != "mode"])
    else:
        inline_query = None

    res.append(
        [
            tg.InlineKeyboardButton(
                text=module.langtable[user.language_code]["cancel-button"],
                callback_data=f"{user.id}\nMAIN\nCANCEL",
            ),
            tg.InlineKeyboardButton(
                text=module.langtable[user.language_code]["play-button"],
                switch_inline_query=inline_query,
                callback_data=None if inline_query else f"{user.id}\nMAIN\nPLAY",
            ),
        ]
    )

    return tg.InlineKeyboardMarkup(res)


def parse_options(keyboard: tg.InlineKeyboardMarkup) -> dict[str, int]:
    res = {}
    for column in keyboard.inline_keyboard[:-1]:
        column_data = parse_callbackquery_data(column[1].callback_data)["args"]
        res[column_data[1]] = column_data[2]

    return res


def get_options_list(indexes: dict[str, int], key: str) -> list[str]:
    res = []
    for option, conditions in module.OPTIONS[key]["options"].items():
        res.append(option)
        for c_option, c_func in conditions.items():
            if not c_func(
                tuple(module.OPTIONS[c_option]["options"])[indexes.get(c_option, 0)]
            ):
                del res[-1]
                break

    return res


def encode_data(self, obj):
    self.bot_data["matches"] = {
        k: v.to_dict() for k, v in self.bot_data["matches"].items()
    }
    res = self.default_encoder(self, obj)
    return {k: gzip.compress(v) for k, v in res.items()}


@avoid_spam
def start(update: tg.Update, context: db_utils.RedisContext):
    update.effective_chat.send_message(text=context.langtable["start-msg"])


@avoid_spam
def unknown(update: tg.Update, context: db_utils.RedisContext):
    ratios = []
    d = difflib.SequenceMatcher(a=update.effective_message.text)
    for command in context.langtable["cmds"].keys():
        d.set_seq2(command)
        ratios.append((d.ratio(), command))
    suggested = max(ratios, key=lambda x: x[0])[1]
    update.effective_message.reply_text(
        context.langtable["unknown-cmd"].format(suggested=suggested)
    )


@avoid_spam
def settings(update: tg.Update, context: db_utils.RedisContext):
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
                        callback_data=f"{update.effective_user.id}\nMAIN\n{'ANON_MODE_OFF' if is_anon else 'ANON_MODE_ON'}",
                    )
                ]
            ]
        ),
    )


@avoid_spam
def boardgame_menu(update: tg.Update, context: db_utils.RedisContext) -> None:
    update.effective_chat.send_message(
        context.langtable["game-setup"],
        reply_markup=format_options(update.effective_user),
    )


@tg_adapter
def send_invite_inline(update: tg.Update, context: db_utils.RedisContext) -> None:
    raw = update.inline_query.query.strip(" ")
    options = (
        token.split(":")
        for token in raw.split(" ")
        if INLINE_OPTION_PATTERN.match(token)
    )
    options = {k: int(v) for k, v in options}
    try:
        challenge_desc = prettify_options(options, update.effective_user.language_code)
    except LookupError:
        return

    match_id = module.create_match_id()

    update.inline_query.answer(
        results=(
            tg.InlineQueryResultPhoto(
                match_id,
                module.INVITE_IMAGE,
                module.INVITE_THUMBNAIL,
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
                                callback_data=f"\nMAIN\nACCEPT#{match_id}",
                            )
                        ]
                    ]
                ),
            ),
        )
    )


@tg_adapter
def create_invite(update: tg.Update, context: db_utils.RedisContext):
    raw = update.chosen_inline_result.query.strip(" ")
    options = (
        token.split(":")
        for token in raw.split(" ")
        if INLINE_OPTION_PATTERN.match(token)
    )
    options = {k: int(v) for k, v in options}

    context.db.create_invite(
        update.chosen_inline_result.result_id, update.effective_user, options
    )


@avoid_spam
def button_callback(
    update: tg.Update, context: db_utils.RedisContext
) -> Optional[tuple]:
    args = parse_callbackquery_data(update.callback_query.data)
    logger.debug(f"Handling user input: {args}")
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
        if args["args"][0] == "NA":
            update.callback_query.answer(text=context.langtable["error-popup-msg"])
        elif args["args"][0] == "ANON_MODE_OFF":
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
                                callback_data=f"{update.effective_user.id}\nMAIN\nANON_MODE_ON",
                            )
                        ]
                    ]
                )
            )
        elif args["args"][0] == "ANON_MODE_ON":
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
                                callback_data=f"{update.effective_user.id}\nMAIN\nANON_MODE_OFF",
                            )
                        ]
                    ]
                )
            )

        elif args["args"][0] == "PREV":
            options = parse_options(update.effective_message.reply_markup)
            if options[args["args"][1]] == 0:
                options[args["args"][1]] = (
                    len(get_options_list(options, args["args"][1])) - 1
                )
            else:
                options[args["args"][1]] -= 1

            update.effective_message.edit_reply_markup(
                format_options(update.effective_user, indexes=options)
            )
            new_value = tuple(module.OPTIONS[args["args"][1]]["options"])[
                options[args["args"][1]]
            ]
            update.callback_query.answer(
                f"{context.langtable['chosen-popup']} {context.langtable[args['args'][1]]} - {context.langtable[new_value]}"
            )

        elif args["args"][0] == "NEXT":
            options = parse_options(update.effective_message.reply_markup)
            if (
                options[args["args"][1]]
                == len(get_options_list(options, args["args"][1])) - 1
            ):
                options[args["args"][1]] = 0
            else:
                options[args["args"][1]] += 1

            update.effective_message.edit_reply_markup(
                format_options(update.effective_user, indexes=options)
            )
            new_value = tuple(module.OPTIONS[args["args"][1]]["options"])[
                options[args["args"][1]]
            ]
            update.callback_query.answer(
                f"{context.langtable['chosen-popup']} {context.langtable[args['args'][1]]} - {context.langtable[new_value]}"
            )

        elif args["args"][0] == "DESC":
            update.callback_query.answer()

        elif args["args"][0] == "PLAY":
            options = parse_options(update.effective_message.reply_markup)
            options = {k: module.OPTIONS[k]["options"][v] for k, v in options.items()}

            if options["mode"] == "vsbot":
                update.effective_message.edit_text(context.langtable["match-found"])
                new = module.AIMatch(
                    update.effective_user,
                    update.effective_chat.id,
                    options=options,
                    bot=context.bot,
                )
                context.bot_data["matches"][new.id] = new
                new.init_turn(update.effective_user.language_code)
            elif options["mode"] == "online":
                opponent = get_opponent(context.bot_data["queue"], options)
                if opponent:
                    context.bot_data["queue"].remove(opponent)
                    update.effective_message.edit_text(context.langtable["match-found"])
                    opponent["msg"].edit_text(context.langtable["match-found"])

                    if opponent["chat_id"] == update.effective_chat.id:
                        new = module.GroupMatch(
                            opponent["user"],
                            update.effective_user,
                            opponent["chat_id"],
                            options=options,
                            bot=context.bot,
                        )
                    else:
                        new = module.PMMatch(
                            opponent["user"],
                            update.effective_user,
                            opponent["chat_id"],
                            update.effective_chat.id,
                            options=options,
                            bot=context.bot,
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
                    msg_lines = [
                        context.langtable["awaiting-opponent"],
                        "",
                        context.langtable["options-title"],
                        ", ".join([context.langtable[i] for i in options.values()]),
                    ]
                    update.effective_message.edit_text(
                        "\n".join(msg_lines),
                        reply_markup=tg.InlineKeyboardMarkup(
                            [
                                [
                                    tg.InlineKeyboardButton(
                                        text=context.langtable["cancel-button"],
                                        callback_data=f"{update.effective_user.id}\nMAIN\nCANCEL",
                                    )
                                ]
                            ]
                        ),
                    )
            elif options["mode"] == "invite":
                update.effective_message.edit_text(context.langtable["invite-sent"])

        elif args["args"][0] == "CANCEL":
            for index, queued in enumerate(context.bot_data["queue"]):
                if queued["user"] == update.effective_user:
                    del context.bot_data["queue"][index]
                    update.effective_message.edit_text(
                        context.langtable["search-cancelled"]
                    )
                    return

            update.callback_query.answer(context.langtable["error-popup-msg"])
            update.effective_message.edit_reply_markup()

        elif args["args"][0] == "ACCEPT":
            challenge = context.db.get_invite(args["args"][1])
            if challenge:
                new = module.GroupMatch(
                    challenge["from_user"],
                    update.effective_user,
                    None,
                    msg=module.InlineMessageAdapter(
                        update.callback_query.inline_message_id, context.bot
                    ),
                    options=challenge["options"],
                    bot=context.bot,
                )
                context.bot_data["matches"][new.id] = new
                new.init_turn()
            else:
                context.bot.edit_message_text(
                    inline_message_id=update.callback_query.inline_message_id,
                    text=context.langtable["error-msg"],
                )

    elif args["target_id"] == "module":
        module.KEYBOARD_BUTTONS[args["args"[0]]](update, context, args["args"])
    else:
        if context.bot_data["matches"].get(args["target_id"]):
            res = context.bot_data["matches"][args["target_id"]].handle_input(args["args"])
            res = res if res else (None, False)
            if context.bot_data["matches"][args["target_id"]].result != "*":
                del context.bot_data["matches"][args["target_id"]]
            update.callback_query.answer(text=res[0], show_alert=res[1])
        else:
            update.callback_query.answer(text=context.langtable["game-not-found-error"])


def create_app() -> flask.Flask:
    dispatcher = tg.ext.Updater(
        token=os.environ["BOT_TOKEN"],
        defaults=tg.ext.Defaults(quote=True),
        persistence=db_utils.RedisPersistence(
            url=os.environ["REDISCLOUD_URL"],
            store_user_data=False,
            store_chat_data=False,
            encoder=encode_data,
            decoder=decode_data,
        ),
        context_types=tg.ext.ContextTypes(context=db_utils.RedisContext),
        user_sig_handler=stop_bot
    ).dispatcher
    if not dispatcher.bot_data:
        dispatcher.bot_data["matches"] = {}
    else:
        dispatcher.bot_data["matches"] = {
            k: module.from_dict(v, k, dispatcher.bot)
            for k, v in dispatcher.bot_data["matches"].items()
        }
    dispatcher.bot_data.update({"challenges": {}, "queue": []})

    module.init(IS_DEBUG, dispatcher.persistence.conn)

    dispatcher.add_handler(tg.ext.InlineQueryHandler(send_invite_inline))
    dispatcher.add_handler(tg.ext.ChosenInlineResultHandler(create_invite))
    dispatcher.add_handler(tg.ext.CallbackQueryHandler(button_callback))
    dispatcher.add_handler(tg.ext.CommandHandler("start", start))
    dispatcher.add_handler(
        tg.ext.CommandHandler(module.__name__, boardgame_menu)
    )
    dispatcher.add_handler(tg.ext.CommandHandler("settings", settings))
    dispatcher.add_handler(
        tg.ext.MessageHandler(tg.ext.filters.Filters.regex("^/"), unknown)
    )
    dispatcher.add_error_handler(error_handler)

    for key in module.langtable.keys():
        dispatcher.bot.set_my_commands(
            list(module.langtable[key]["cmds"].items()), language_code=key
        )

    app = flask.Flask(__name__)

    def process_update():
        logger.info(f"Handling update: {flask.request.json}")
        dispatcher.process_update(tg.Update.de_json(flask.request.json, dispatcher.bot))
        return "<p>True</p>"

    def fetch_dynamic(filename):
        path = os.path.join("images", "temp", filename)
        data = io.BytesIO(open(path, "rb").read())
        os.remove(path)
        return flask.send_file(data, download_name=filename)

    def fetch_static(filename):
        return flask.send_file(
            os.path.join("images", "static", filename), 
            download_name=filename
        )

    app.add_url_rule(f"/{os.environ['BOT_TOKEN']}/dynamic/<filename>", view_func=fetch_dynamic)
    app.add_url_rule(f"/{os.environ['BOT_TOKEN']}/static/<filename>", view_func=fetch_static)
    app.add_url_rule(f"/{os.environ['BOT_TOKEN']}/send-update", view_func=process_update, methods=["POST"])
    dispatcher.bot.set_webhook("/".join([os.environ["HOST_URL"], os.environ["BOT_TOKEN"], "send-update"]))
    dispatcher.bot.send_message(chat_id=os.environ["CREATOR_ID"], text="Бот включен")

    return app


if __name__ == "__main__":
    from pyngrok import ngrok

    os.environ["HOST_URL"] = ngrok.connect(addr="4096").public_url.replace("http://", "https://")
    
    app = create_app()
    app.run(host="localhost", port=4096)
