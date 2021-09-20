import os
import json
import telegram as tg
import chess
import db_utils
import difflib
import gzip
import logging

logging.basicConfig(
    format="%(relativeCreated)s %(module)s %(message)s", level=logging.DEBUG
)

if os.path.exists("debug_env.json"):
    chess.BaseMatch.ENGINE_FILENAME = "./stockfish"
    with open("debug_env.json") as r:
        os.environ.update(json.load(r))

try:
    os.mkdir(os.path.join("images", "temp"))
except FileExistsError:
    pass

langtable = json.load(open("langtable.json"))

group_thread = tg.ext.DelayQueue()
pm_thread = tg.ext.DelayQueue()


def avoid_spam(f):
    def decorated(update: tg.Update, context: db_utils.RedisContext):
        context.db.sadd("user-ids", str(update.effective_user.id).encode())
        cur_lang_reqcount = context.db.get(
            f"lang-reqcount:{update.effective_user.language_code}"
        )
        cur_lang_reqcount = int(cur_lang_reqcount) if cur_lang_reqcount else 0
        context.db.set(
            f"lang-reqcount:{update.effective_user.language_code}",
            cur_lang_reqcount + 1,
        )
        context.langtable = langtable[update.effective_user.language_code]

        if update.effective_chat.type == "private":
            pm_thread._queue.put((f, (update, context), {}))
        else:
            group_thread._queue.put((f, (update, context), {}))

    return decorated


def stop_bot(*args):
    stats_text = "Запросы боту по языковым кодам:\n"
    for key in updater.persistence.conn.keys(pattern="lang-reqcount:*"):
        stats_text += (
            f"\n{key[:key.find(b':') + 1]}: {updater.persistence.conn.get(key)}"
        )
        updater.persistence.conn.delete(key)
    updater.bot.send_message(chat_id=os.environ["CREATOR_ID"], text=stats_text)
    group_thread.stop()
    pm_thread.stop()
    exit()


def decode_data(self, obj):
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


def encode_data(self, obj):
    self.bot_data["matches"] = {
        k: v.to_dict() for k, v in self.bot_data["matches"].items()
    }
    res = self.default_encoder(self, obj)
    return {k: gzip.compress(v) for k, v in res.items()}


updater = tg.ext.Updater(
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
    user_sig_handler=stop_bot,
)
if not updater.dispatcher.bot_data:
    updater.dispatcher.bot_data = {"queue": [], "matches": {}}
else:
    updater.dispatcher.bot_data["matches"] = {
        k: chess.from_dict(v, k, updater.bot)
        for k, v in updater.dispatcher.bot_data["matches"].items()
    }
chess.BaseMatch.db = updater.persistence.conn


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
    is_anon = context.db.is_anon(update.effective_user)
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
def boardgame_menu(update: tg.Update, context: db_utils.RedisContext):
    keyboard = tg.InlineKeyboardMarkup(
        [
            [
                tg.InlineKeyboardButton(
                    text=context.langtable[i["text"]],
                    callback_data=f"{update.effective_user.id}\nMAIN\nNEW#{i['code']}",
                )
                for i in chess.MODES
            ],
            [
                tg.InlineKeyboardButton(
                    text=context.langtable["cancel-button"],
                    callback_data=f"{update.effective_user.id}\nMAIN\nCANCEL#",
                )
            ],
        ]
    )
    update.effective_message.reply_text(context.langtable["choosing-gamemode"], reply_markup=keyboard)


@avoid_spam
def button_callback(update: tg.Update, context: db_utils.RedisContext):
    args = parse_callbackquery_data(update.callback_query.data)
    logging.debug(f"Handling user input: {args}")
    if (
        args["expected_uid"]
        and args["expected_uid"] != update.callback_query.from_user.id
    ):
        if args["target_id"] == "MAIN":
            update.callback_query.answer(context.langtable["error-popup-msg"])
        else:
            update.callback_query.answer(
                text=context.langtable["chess-unexpected-uid"],
                show_alert=True,
            )
        return

    if args["target_id"] == "MAIN":
        if args["args"][0] == "NA":
            update.callback_query.answer(text=context.langtable["error-popup-msg"])
        elif args["args"][0] == "DOWNLOAD":
            content = context.db.get(f"{args['args'][1]}:pgn")
            update.effective_message.edit_reply_markup()
            if content:
                update.effective_message.reply_document(
                    gzip.decompress(content),
                    caption=context.langtable["pgn-file-caption"],
                    filename=f"chess4u-{args['args'][1]}.pgn",
                )
                context.db.delete(f"{args['args'][1]}:pgn")
            else:
                update.callback_query.answer(
                    context.langtable["pgn-fetch-error"], show_alert=True
                )
        elif args["args"][0] == "ANON_MODE_OFF":
            context.db.anon_mode_off(update.effective_user)
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
                                    if context.db.is_anon(update.effective_user)
                                    else "🔴"
                                ),
                                callback_data=f"{update.effective_user.id}\nMAIN\nANON_MODE_ON",
                            )
                        ]
                    ]
                )
            )
        elif args["args"][0] == "ANON_MODE_ON":
            context.db.anon_mode_on(update.effective_user)
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
                                    if context.db.is_anon(update.effective_user)
                                    else "🔴"
                                ),
                                callback_data=f"{update.effective_user.id}\nMAIN\nANON_MODE_OFF",
                            )
                        ]
                    ]
                )
            )
        elif args["args"][0] == "NEW":
            update.effective_message.edit_text(context.langtable["match-found"])
            if args["args"][1] == "AI":
                new = chess.AIMatch(
                    update.effective_user, update.effective_chat.id, bot=context.bot
                )
                context.bot_data["matches"][new.id] = new
                new.init_turn(update.effective_user.language_code, setup=True)

            elif len(context.bot_data["queue"]) > 0:
                queued_user, queued_chat, queued_msg = context.bot_data["queue"].pop(0)
                if queued_chat == update.effective_chat:
                    new = chess.GroupMatch(
                        queued_user,
                        update.effective_user,
                        update.effective_chat.id,
                        bot=context.bot,
                    )
                else:
                    new = chess.PMMatch(
                        queued_user,
                        update.effective_user,
                        queued_chat.id,
                        update.effective_chat.id,
                        bot=context.bot,
                    )
                queued_msg.edit_text(context.langtable["match-found"])
                context.bot_data["matches"][new.id] = new
                new.init_turn(update.effective_user.language_code)
            else:
                keyboard = tg.InlineKeyboardMarkup(
                    [
                        [
                            tg.InlineKeyboardButton(
                                text=context.langtable["cancel-button"],
                                callback_data=f"{update.effective_user.id}\nMAIN\nCANCEL#$u",
                            )
                        ]
                    ]
                )
                update.effective_message.edit_text(
                    text=context.langtable["awaiting-opponent"], reply_markup=keyboard
                )
                context.bot_data["queue"].append(
                    (
                        update.effective_user,
                        update.effective_chat,
                        update.effective_message,
                    )
                )
        elif args["args"][0] == "CANCEL":
            for index, queued in enumerate(context.bot_data["queue"]):
                if queued[0].id == args["args"][1]:
                    queued[2].edit_text(context.langtable["search-cancelled"])
                    del context.bot_data["queue"][index]

    else:
        if context.bot_data["matches"].get(args["target_id"]):
            res = context.bot_data["matches"][args["target_id"]].handle_input(
                update.effective_user.language_code,
                args["args"],
            )
            res = res if res else (None, False)
            if context.bot_data["matches"][args["target_id"]].result != "*":
                del context.bot_data["matches"][args["target_id"]]
            update.callback_query.answer(text=res[0], show_alert=res[1])
        else:
            update.callback_query.answer(text=context.langtable["game-not-found-error"])


def main():
    updater.dispatcher.add_handler(tg.ext.CallbackQueryHandler(button_callback))
    updater.dispatcher.add_handler(tg.ext.CommandHandler("start", start))
    updater.dispatcher.add_handler(tg.ext.CommandHandler("chess", boardgame_menu))
    updater.dispatcher.add_handler(tg.ext.CommandHandler("settings", settings))
    updater.dispatcher.add_handler(
        tg.ext.MessageHandler(tg.ext.filters.Filters.regex("^/"), unknown)
    )
    for key in langtable.keys():
        updater.bot.set_my_commands(
            list(langtable[key]["cmds"].items()), language_code=key
        )

    updater.bot.send_message(chat_id=os.environ["CREATOR_ID"], text="Бот включен")
    updater.start_polling(drop_pending_updates=True)
    updater.idle()


if __name__ == "__main__":
    main()
