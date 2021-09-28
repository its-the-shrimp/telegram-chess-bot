import os
import json
from typing import Optional
import telegram as tg
import chess as module
import db_utils
import difflib
import gzip
import logging
import collections

if os.path.exists(os.path.join(os.path.dirname(__file__), "debug_env.json")):
    loglevel = logging.DEBUG
    module.BaseMatch.ENGINE_FILENAME = "./stockfish"
    with open(
        os.path.join(os.path.join(os.path.dirname(__file__), "debug_env.json"))
    ) as r:
        os.environ.update(json.load(r))
else:
    loglevel = logging.INFO

logging.basicConfig(
    format="%(relativeCreated)s %(module)s %(message)s", level=logging.DEBUG
)

try:
    os.mkdir(os.path.join(os.path.dirname(__file__), "images", "temp"))
except FileExistsError:
    pass


group_thread = tg.ext.DelayQueue()
pm_thread = tg.ext.DelayQueue()


class QueuePoint:
    def __init__(
        self,
        user: tg.User,
        msg: tg.Message,
        invite_msg: tg.Message = None,
        options: dict[str, str] = {},
    ):
        self.user = user
        self.chat_id = msg.chat_id
        self.msg = msg
        self.invite_msg = invite_msg
        self.options = options
        self.is_user_invite = invite_msg and self.chat_id != invite_msg.chat_id
        self.is_chat_invite = invite_msg and self.chat_id == invite_msg.chat_id


def get_opponent(
    queue: list[QueuePoint],
    user: tg.User,
    chat_id: int,
    options: dict = {},
    created_by_uid: int = None,
) -> Optional[QueuePoint]:
    opponent, invite = (None, None)
    for index in range(len(queue) - 1, -1, -1):
        queuepoint = queue[index]
        if all(
            [
                options == queuepoint.options,
                chat_id == queuepoint.chat_id or not queuepoint.is_chat_invite,
                not queuepoint.is_user_invite
                or user.id == queuepoint.invite_msg.chat_id,
                created_by_uid == queuepoint.user.id or created_by_uid is None,
            ]
        ):
            if queuepoint.is_chat_invite or queuepoint.is_user_invite:
                invite = index
            else:
                opponent = index

    if invite is not None:
        return queue.pop(invite)
    elif opponent is not None:
        return queue.pop(opponent)


def avoid_spam(f):
    def decorated(update: tg.Update, context: db_utils.RedisContext):
        context.db.cache_user_data(update.effective_user)
        context.langtable = module.langtable[update.effective_user.language_code]

        if update.effective_chat.type == "private":
            pm_thread._queue.put((f, (update, context), {}))
        else:
            group_thread._queue.put((f, (update, context), {}))

    return decorated


def stop_bot(_, frame):
    updater = frame.f_locals["self"]
    counter = collections.defaultdict(int)
    for key in updater.persistence.conn.keys(pattern="*:lang"):
        counter[updater.persistence.conn.get(key).decode()] += 1
    updater.bot.send_message(
        text="\n".join([f"{k}: {v}" for k, v in counter.items()]),
        chat_id=os.environ["CREATOR_ID"],
    )
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


def format_options(
    user: tg.User, indexes: dict[str, int] = {}
) -> tg.InlineKeyboardMarkup:
    res = []
    for name, option in module.OPTIONS.items():
        chosen = option["options"][indexes.get(name, 0)]
        conditions_passed = True
        for c_option, c_func in option["conditions"].items():
            if not c_func(
                module.OPTIONS[c_option]["options"][indexes.get(c_option, 0)]
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

    res.append(
        [
            tg.InlineKeyboardButton(
                text=module.langtable[user.language_code]["cancel-button"],
                callback_data=f"{user.id}\nMAIN\nCANCEL",
            ),
            tg.InlineKeyboardButton(
                text=module.langtable[user.language_code]["play-button"],
                callback_data=f"{user.id}\nMAIN\nPLAY",
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
def boardgame_menu(update: tg.Update, context: db_utils.RedisContext) -> None:
    context.user_data["pending_options"] = None
    update.effective_message.reply_text(
        context.langtable["game-setup"],
        reply_markup=format_options(update.effective_user),
    )


@avoid_spam
def spec_invite_with_contact(update: tg.Update, context: db_utils.RedisContext) -> None:
    if "pending_options" in context.user_data:
        opponent_langtable = module.langtable[
            context.db.get_langcode(update.effective_message.contact.user_id)
        ]
        invite_msg = context.bot.send_message(
            update.message.contact.user_id,
            opponent_langtable["invite-msg"].format(
                name=update.effective_user.name,
                options=", ".join(
                    [
                        opponent_langtable[v]
                        for k, v in context.user_data["pending_options"].items()
                        if k != "mode"
                    ]
                ),
            ),
            reply_markup=tg.InlineKeyboardMarkup(
                [
                    [
                        tg.InlineKeyboardButton(
                            text=opponent_langtable["accept-button"],
                            callback_data=f"{update.effective_message.contact.user_id}\nMAIN\nACCEPT#{update.effective_user.id}",
                        ),
                        tg.InlineKeyboardButton(
                            text=opponent_langtable["decline-button"],
                            callback_data=f"{update.effective_message.contact.user_id}\nMAIN\nDECLINE#{update.effective_user.id}",
                        ),
                    ],
                ]
            ),
        )
        context.bot_data["queue"].append(
            QueuePoint(
                update.effective_user,
                context.user_data["pending_req_msg"],
                invite_msg=invite_msg,
                options=context.user_data["pending_options"],
            )
        )


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
                text=context.langtable["unexpected-uid"],
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

        elif args["args"][0] == "PREV":
            options = parse_options(update.effective_message.reply_markup)
            if options[args["args"][1]] == 0:
                options[args["args"][1]] = (
                    len(module.OPTIONS[args["args"][1]]["options"]) - 1
                )
            else:
                options[args["args"][1]] -= 1

            update.effective_message.edit_reply_markup(
                format_options(update.effective_user, indexes=options)
            )
            new_value = module.OPTIONS[args["args"][1]]["options"][
                options[args["args"][1]]
            ]
            update.callback_query.answer(context.langtable[new_value])

        elif args["args"][0] == "NEXT":
            options = parse_options(update.effective_message.reply_markup)
            if (
                options[args["args"][1]]
                == len(module.OPTIONS[args["args"][1]]["options"]) - 1
            ):
                options[args["args"][1]] = 0
            else:
                options[args["args"][1]] += 1

            update.effective_message.edit_reply_markup(
                format_options(update.effective_user, indexes=options)
            )
            new_value = module.OPTIONS[args["args"][1]]["options"][
                options[args["args"][1]]
            ]
            update.callback_query.answer(context.langtable[new_value])

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
                opponent = get_opponent(
                    context.bot_data["queue"],
                    update.effective_user,
                    update.effective_chat.id,
                    options=options,
                )
                if opponent:
                    update.effective_message.edit_text(context.langtable["match-found"])
                    opponent.msg.edit_text(context.langtable["match-found"])

                    if opponent.chat_id == update.effective_chat.id:
                        new = module.GroupMatch(
                            opponent.user,
                            update.effective_user,
                            opponent.chat_id,
                            options=options,
                            bot=context.bot,
                        )
                    else:
                        new = module.PMMatch(
                            opponent.user,
                            update.effective_user,
                            opponent.chat_id,
                            update.effective_chat.id,
                            options=options,
                            bot=context.bot,
                        )
                    context.bot_data["matches"][new.id] = new
                    new.init_turn(update.effective_user.language_code)
                else:
                    context.bot_data["queue"].append(
                        QueuePoint(
                            update.effective_user,
                            update.effective_message,
                            options=options,
                        )
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
                context.user_data.update(
                    {
                        "pending_req_msg": update.effective_message.edit_text(
                            context.langtable["invite-spec-req"]
                        ),
                        "pending_options": options,
                    }
                )

        elif args["args"][0] == "CANCEL":
            for index, queued in enumerate(context.bot_data["queue"]):
                if queued.user == update.effective_user:
                    del context.bot_data["queue"][index]
                    update.effective_message.edit_text(
                        context.langtable["search-cancelled"]
                    )
                    return

            update.callback_query.answer(context.langtable["error-popup-msg"])
            update.effective_message.edit_reply_markup()

        elif args["args"][0] == "ACCEPT":
            invite = get_opponent(
                context.bot_data["queue"],
                update.effective_user,
                update.effective_chat.id,
                options=context.dispatcher.user_data[args["args"][1]][
                    "pending_options"
                ],
                created_by_uid=args["args"][1],
            )
            if invite:
                update.effective_message.edit_text(
                    context.langtable["invite-accepted"].format(name=invite.user.name)
                )
                invite.msg.edit_text(context.langtable["match-found"])
                new = module.PMMatch(
                    invite.user,
                    update.effective_user,
                    invite.chat_id,
                    update.effective_chat.id,
                    options=invite.options,
                    bot=context.bot,
                )
                context.bot_data["matches"][new.id] = new
                new.init_turn(update.effective_user.language_code)

        elif args["args"][0] == "DECLINE":
            for index, queued in enumerate(context.bot_data["queue"]):
                if all(
                    [
                        queued.is_user_invite
                        and queued.invite_msg.chat_id == update.effective_user,
                        queued.user.id == args["args"][1],
                    ]
                ):
                    del context.bot_data["queue"][index]
                    update.effective_message.edit_text(
                        context.langtable["invite-declined"]
                    )
                    return

            update.callback_query.answer(context.langtable["error-popup-msg"])
            update.effective_message.edit_reply_markup()

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
            k: module.from_dict(v, k, updater.bot)
            for k, v in updater.dispatcher.bot_data["matches"].items()
        }
    module.BaseMatch.db = updater.persistence.conn

    updater.dispatcher.add_handler(tg.ext.CallbackQueryHandler(button_callback))
    updater.dispatcher.add_handler(tg.ext.CommandHandler("start", start))
    updater.dispatcher.add_handler(
        tg.ext.CommandHandler(module.__name__, boardgame_menu)
    )
    updater.dispatcher.add_handler(tg.ext.CommandHandler("settings", settings))
    updater.dispatcher.add_handler(
        tg.ext.MessageHandler(tg.ext.filters.Filters.contact, spec_invite_with_contact)
    )
    updater.dispatcher.add_handler(
        tg.ext.MessageHandler(tg.ext.filters.Filters.regex("^/"), unknown)
    )
    for key in module.langtable.keys():
        updater.bot.set_my_commands(
            list(module.langtable[key]["cmds"].items()), language_code=key
        )

    updater.bot.send_message(chat_id=os.environ["CREATOR_ID"], text="Бот включен")
    updater.start_polling(drop_pending_updates=True)
    updater.idle()


if __name__ == "__main__":
    main()
