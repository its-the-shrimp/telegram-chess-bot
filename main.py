import os
import os.path
import json
import telegram as tg
import telegram.ext
import random
import chess
import bot_utils
import difflib
import gzip

if os.path.exists("debug_env.json"):
    import logging

    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)

    with open("debug_env.json") as r:
        os.environ.update(json.load(r))

try:
    os.mkdir(os.path.join("images", "temp"))
except FileExistsError:
    pass

group_thread = tg.ext.DelayQueue()
pm_thread = tg.ext.DelayQueue()


def avoid_spam(f):
    def decorated(update, context):
        context.db.sadd("user-ids", str(update.effective_user.id).encode())
        if update.effective_chat.type == "private":
            pm_thread._queue.put((f, (update, context), {}))
        else:
            group_thread._queue.put((f, (update, context), {}))

    return decorated


def stop_bot(*args):
    group_thread.stop()
    pm_thread.stop()
    exit()


def decode_data(self, obj):
    return {k: gzip.decompress(v).decode() for k, v in obj.items()}

def parse_callbackquery_data(data: str) -> dict:
    data = data.split('\n')
    res = {'expected_uid': int(data[0]) if data[0] else None, 'target_id': data[1], 'args': []}
    for argument in data[2].split('#'):
        if argument.isdigit():
            res['args'].append(int(argument))
        elif argument == '$u':
            res['args'].append(res['expected_uid'])
        elif argument:
            res['args'].append(argument)
        else:
            res['args'].append(None)
    
    return res

def encode_data(self, obj):
    self.bot_data["matches"] = {
        k: v.to_dict() for k, v in self.bot_data["matches"].items()
    }
    res = self.default_encoder(self, obj)
    return {k: gzip.compress(v) for k, v in res.items()}


commands = [
    ("/play", "Играть в шахматы"),
    ("/settings", "Настройки бота, такие как анонимный режим и др."),
]
updater = tg.ext.Updater(
    token=os.environ["BOT_TOKEN"],
    defaults=tg.ext.Defaults(quote=True),
    persistence=bot_utils.RedisPersistence(
        url=os.environ["REDISCLOUD_URL"],
        store_user_data=False,
        store_chat_data=False,
        encoder=encode_data,
        decoder=decode_data,
    ),
    context_types=tg.ext.ContextTypes(context=bot_utils.RedisContext),
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
def start(update, context):
    update.effective_chat.send_message(
        text="Привет! Чтобы сыграть, введи команду /play"
    )


@avoid_spam
def unknown(update, context):
    ratios = []
    d = difflib.SequenceMatcher(a=update.effective_message.text)
    for command, _ in commands:
        d.set_seq2(command)
        ratios.append((d.ratio(), command))
    suggested = max(ratios, key=lambda x: x[0])[1]
    update.effective_message.reply_text(
        f"Неизвестная команда. Может, вы имели ввиду {suggested}?"
    )


@avoid_spam
def settings(update, context):
    is_anon = context.db.is_anon(update.effective_user)
    update.effective_message.reply_text(
        """
Опции:
    <i>Анонимный режим</i>: Бот не будет оставлять ваше имя пользователя (начинающееся с @)
        в сообщениях и во вложениях к ним.
    """,
        parse_mode=tg.ParseMode.HTML,
        reply_markup=tg.InlineKeyboardMarkup(
            [
                [
                    tg.InlineKeyboardButton(
                        text=f'Анонимный режим: {"🟢" if is_anon else "🔴"}',
                        callback_data=f"{update.effective_user.id}\nMAIN\n{'ANON_MODE_OFF' if is_anon else 'ANON_MODE_ON'}",
                    )
                ]
            ]
        ),
    )


@avoid_spam
def boardgame_menu(update, context):
    keyboard = tg.InlineKeyboardMarkup(
        [
            [
                tg.InlineKeyboardButton(
                    text=i["text"],
                    callback_data=f"{update.effective_user.id}\nMAIN\nNEW#{i['code']}",
                )
                for i in chess.MODES
            ],
            [tg.InlineKeyboardButton(text='Отменить', callback_data=f"{update.effective_user.id}\nMAIN\nCANCEL#")]
        ]
    )
    update.effective_message.reply_text("Выберите режим:", reply_markup=keyboard)


@avoid_spam
def button_callback(update, context):
    args = parse_callbackquery_data(update.callback_query.data)
    print(args)
    if args["expected_uid"] and args["expected_uid"] != update.callback_query.from_user.id:
        if args["target_id"] == "MAIN":
            update.callback_query.answer("Ошибка")
        else:
            update.callback_query.answer(
                text=context.bot_data["matches"][args["target_id"]].WRONG_PERSON_MSG,
                show_alert=True,
            )
        return

    if args["target_id"] == "MAIN":
        if args["args"][0] == "NA":
            update.callback_query.answer(text="Недоступно", show_alert=True)
        elif args["args"][0] == "ANON_MODE_OFF":
            context.db.anon_mode_off(update.effective_user)
            update.callback_query.answer("Анонимный режим отключен", show_alert=True)
            update.effective_message.edit_reply_markup(
                tg.InlineKeyboardMarkup(
                    [
                        [
                            tg.InlineKeyboardButton(
                                text=f'Анонимный режим: {"🟢" if context.db.is_anon(update.effective_user) else "🔴"}',
                                callback_data=f"{update.effective_user.id}\nMAIN\nANON_MODE_ON",
                            )
                        ]
                    ]
                )
            )
            context.drop_callback_data(update.callback_query)
        elif args["args"][0] == "ANON_MODE_ON":
            context.db.anon_mode_on(update.effective_user)
            update.callback_query.answer("Анонимный режим включен", show_alert=True)
            update.effective_message.edit_reply_markup(
                tg.InlineKeyboardMarkup(
                    [
                        [
                            tg.InlineKeyboardButton(
                                text=f'Анонимный режим: {"🟢" if context.db.is_anon(update.effective_user) else "🔴"}',
                                callback_data=f"{update.effective_user.id}\nMAIN\nANON_MODE_OFF",
                            )
                        ]
                    ]
                )
            )
            context.drop_callback_data(update.callback_query)
        elif args["args"][0] == "NEW":
            if args["args"][1] == "AI":
                update.effective_message.edit_text("Игра найдена")
                new = chess.AIMatch(
                    update.effective_user, update.effective_chat.id, bot=context.bot
                )
                context.bot_data["matches"][new.id] = new
                new.init_turn(setup=True)

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
                queued_msg.edit_text("Игра найдена")
                update.effective_message.edit_text(text="Игра найдена")
                context.bot_data["matches"][new.id] = new
                new.init_turn()
            else:
                keyboard = tg.InlineKeyboardMarkup(
                    [
                        [
                            tg.InlineKeyboardButton(
                                text="Отмена",
                                callback_data=f"{update.effective_user.id}\nMAIN\nCANCEL#$u",
                            )
                        ]
                    ]
                )
                update.effective_message.edit_text(
                    text="Ждём игроков...", reply_markup=keyboard
                )
                context.bot_data["queue"].append(
                    (
                        update.effective_user,
                        update.effective_chat,
                        update.effective_message,
                    )
                )
        elif args["args"][0] == "CANCEL":
            if args.get('uid'):
                for index, queued in enumerate(context.bot_data["queue"]):
                    if queued[0].id == args["args"][1]:
                        queued[2].edit_text("Поиск игры отменен")
                        del context.bot_data["queue"][index]
            else:
                update.callback_query.message.edit_text("Поиск игры отменен")

    else:

        res = context.bot_data["matches"][args["target_id"]].handle_input(args["args"])
        res = res if res else (None, False)
        if context.bot_data["matches"][args["target_id"]].finished:
            del context.bot_data["matches"][args["target_id"]]
        update.callback_query.answer(text=res[0], show_alert=res[1])


def main():
    updater.dispatcher.add_handler(tg.ext.CallbackQueryHandler(button_callback))
    updater.dispatcher.add_handler(tg.ext.CommandHandler("start", start))
    updater.dispatcher.add_handler(tg.ext.CommandHandler("play", boardgame_menu))
    updater.dispatcher.add_handler(tg.ext.CommandHandler("settings", settings))
    updater.dispatcher.add_handler(
        tg.ext.MessageHandler(tg.ext.filters.Filters.regex("^/"), unknown)
    )
    updater.bot.set_my_commands(commands)

    updater.bot.send_message(chat_id=os.environ["CREATOR_ID"], text="Бот включен")
    updater.start_polling(drop_pending_updates=True)
    updater.idle()


if __name__ == "__main__":
    main()
