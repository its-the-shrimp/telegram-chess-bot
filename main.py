import os
import json
#ENV VARIABLES IMPORT FOR DEBUGGING, DO NOT DEPLOY THE FOLLOWING:
with open('debug_env.json') as r:
    env_dict = list(json.loads(r.read()).items())
    for k, v in env_dict:
        os.environ[k] = v
#REMOVE UP TO HERE
import telegram as tg
import telegram.ext as tg_ext
import random
import boardgame_api

group_thread = tg_ext.DelayQueue(burst_limit = 20)
pm_thread = tg_ext.DelayQueue()

def avoid_spam(f):
    def decorated(update, context):
        if update.effective_chat.type == 'private':
            pm_thread._queue.put((f, (update, context), {}))
        else:
            group_thread._queue.put((f, (update, context), {}))
        
    return decorated

def update_ui(bot, data, match_id):
    msg_ids = []
    for msg in data:
        reply_markup = []
        for row in msg.get('answers'):
            reply_markup.append([tg.InlineKeyboardButton(text = i['text'],
                                                      callback_data = {'target_id': match_id,
                                                                       'expected_uid': msg.get('expected_uid'),
                                                                       'args': i['callback_data']}) for i in row])
        reply_markup = tg.InlineKeyboardMarkup(reply_markup) if reply_markup else None
        
        if msg.get('msg_id'):
            if msg.get('img'):
                bot.edit_message_media(chat_id = msg['chat_id'], message_id = msg['msg_id'],
                                       media = tg.InputMediaPhoto(msg['img'], caption = msg.get('text', '')),
                                       reply_markup = reply_markup)
            else:
                bot.edit_message_text(chat_id = msg['chat_id'], message_id = msg['msg_id'],
                                      text = msg['text'],
                                      reply_markup = reply_markup)
                
        else:
            if msg.get('img'):
                msg = bot.send_photo(msg['chat_id'], msg['img'], caption = msg.get('text', ''), reply_markup = reply_markup)
            else:
                msg = bot.send_message(msg['chat_id'], msg['text'], reply_markup = reply_markup)
                
            msg_ids.append(msg.message_id)
    return msg_ids
            
defaults = tg_ext.Defaults(quote = True)
updater = tg_ext.Updater(token=os.environ["BOT_TOKEN"], use_context=True, defaults = defaults,
                         arbitrary_callback_data = True)
updater.dispatcher.bot_data = {'queue': {'chess': []}, 'matches': {}}

@avoid_spam
def start(update, context):
    update.effective_chat.send_message(text = 'Привет! Чтобы сыграть в шахматы и не только, введи команду /play')
updater.dispatcher.add_handler(tg_ext.CommandHandler('start', start))

@avoid_spam
def boardgame_menu(update, context):
    update.effective_message.reply_text(text = 'Выберите игру:',
        reply_markup = tg.InlineKeyboardMarkup([[tg.InlineKeyboardButton(text = 'Шахматы',
                                                                         callback_data = {'target_id': 'MAIN',
                                                                                          'command': 'NEW',
                                                                                          'game': 'chess',
                                                                                          'expected_uid': update.effective_user.id})],
                                               [tg.InlineKeyboardButton(text = 'Нарды',
                                                                        callback_data = {'target_id': 'MAIN',
                                                                                          'command': 'NA',
                                                                                          'game': 'rps',
                                                                                          'expected_uid': update.effective_user.id})]]))
updater.dispatcher.add_handler(tg_ext.CommandHandler('play', boardgame_menu))

@avoid_spam
def button_callback(update, context, data = None):
    args = data if data else update.callback_query.data
    print(args)
    if not data and args['expected_uid'] != update.callback_query.from_user.id:
        if args['target_id'] == 'MAIN':
            update.callback_query.answer()
        else:
            update.callback_query.answer(text = context.bot_data['matches'][args['target_id']].WRONG_PERSON_MSG,
                                         show_alert = True)
        return
    
    if args['target_id'] == 'MAIN':
        if args['command'] == 'NA':
            update.callback_query.answer(text = 'Недоступно', show_alert = True)
        elif args['command'] == 'NEW':
            if len(context.bot_data['queue'][args['game']]) > 0:
                queued_user, queued_chat, queued_msg = context.bot_data['queue'][args['game']].pop(0)
                if queued_chat == update.effective_chat:
                    new = getattr(boardgame_api, args['game']).GroupMatch(queued_user,
                                                                     update.effective_user,
                                                                     update.effective_chat)
                else:
                    new = getattr(boardgame_api, args['game']).PMMatch(queued_user,
                                                                       update.effective_user,
                                                                       queued_chat,
                                                                       update.effective_chat)
                queued_msg.edit_text(text = 'Игра найдена')
                update.effective_message.edit_text(text = 'Игра найдена')
                context.bot_data['matches'][new.id] = new
                msg_ids = update_ui(context.bot, new.init_turn(), new.id)
                context.bot_data['matches'][new.id].ids = msg_ids
            else:
                keyboard = tg.InlineKeyboardMarkup([[tg.InlineKeyboardButton(text = 'Отмена', 
                                                                    callback_data = {'target_id': 'MAIN',
                                                                                    'command': 'CANCEL',
                                                                                    'game': args['game'],
                                                                                    'uid': update.effective_user.id,
                                                                                    'expected_uid': update.effective_user.id})]])
                update.effective_message.edit_text(text = 'Ждём игроков...', reply_markup = keyboard)
                context.bot_data['queue'][args['game']].append((update.effective_user,
                                                                update.effective_chat,
                                                                update.effective_message))
        elif args['command'] == 'CANCEL':
            for index, queued in enumerate(context.bot_data['queue'][args['game']]):
                if queued[0].id == args['uid']:
                    queued[2].edit_text(text = 'Поиск игры отменен')
                    del context.bot_data['queue'][args['game']][index]
                
    else:
        res = context.bot_data['matches'][args['target_id']].handle_input(args['args'])
        msg_ids = update_ui(context.bot, res, args['target_id'])
        context.bot_data['matches'][args['target_id']].ids += msg_ids
        if context.bot_data['matches'][args['target_id']].finished:
            del context.bot_data['matches'][args['target_id']]
        update.callback_query.answer()
updater.dispatcher.add_handler(tg_ext.CallbackQueryHandler(button_callback))
            
updater.bot.send_message(chat_id=os.environ['CREATOR_ID'], text='Бот включен')
updater.start_polling(drop_pending_updates=True)
updater.idle()