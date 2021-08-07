import os
import os.path
import json
import telegram as tg
import telegram.ext as tg_ext
import random
import boardgame_api

if os.path.exists('debug_env.json'):
    with open('debug_env.json') as r:
        os.environ.update(json.load(r))

group_thread = tg_ext.DelayQueue()
pm_thread = tg_ext.DelayQueue()

def avoid_spam(f):
    def decorated(update, context):
        if update.effective_chat.type == 'private':
            pm_thread._queue.put((f, (update, context), {}))
        else:
            group_thread._queue.put((f, (update, context), {}))
        
    return decorated
            
defaults = tg_ext.Defaults(quote = True)
updater = tg_ext.Updater(token=os.environ["BOT_TOKEN"], use_context=True, defaults = defaults,
                         arbitrary_callback_data = True)
updater.dispatcher.bot_data = {'queue': {'chess': []}, 'matches': {}}

@avoid_spam
def start(update, context):
    update.effective_chat.send_message(text = 'Привет! Чтобы сыграть, введи команду /play')
updater.dispatcher.add_handler(tg_ext.CommandHandler('start', start))

@avoid_spam
def boardgame_menu(update, context):
    keyboard = tg.InlineKeyboardMarkup([
        [tg.InlineKeyboardButton(text = i['text'], callback_data = {'target_id': 'MAIN',
                    'command': 'NEW',
                    'expected_uid': update.effective_user.id,
                    'game': 'chess',
                    'mode': i['code']})] for i in getattr(boardgame_api, 'chess').MODES])
    update.effective_message.reply_text('Выберите режим:', reply_markup = keyboard)
updater.dispatcher.add_handler(tg_ext.CommandHandler('play', boardgame_menu))

@avoid_spam
def button_callback(update, context):
    args = update.callback_query.data
    
    if type(args) == tg_ext.InvalidCallbackData:
        update.callback_query.answer('Ошибка: информация о сообщении не найдена.')
        return

    if args['expected_uid'] != update.callback_query.from_user.id:
        if args['target_id'] == 'MAIN':
            update.callback_query.answer()
        else:
            update.callback_query.answer(text = context.bot_data['matches'][args['target_id']].WRONG_PERSON_MSG, show_alert = True)
        return
    
    if args['target_id'] == 'MAIN':
        if args['command'] == 'NA':
            update.callback_query.answer(text = 'Недоступно', show_alert = True)
        elif args['command'] == 'CHOOSE_MODE':
            keyboard = tg.InlineKeyboardMarkup([
                [tg.InlineKeyboardButton(text = i['text'], callback_data = {'target_id': 'MAIN',    
                            'command': 'NEW',
                            'expected_uid': update.effective_user.id,
                            'game': args['game'],
                            'mode': i['code']})] for i in getattr(boardgame_api, args['game']).MODES
            ])
            update.effective_message.edit_text(text = 'Выберите режим:', reply_markup = keyboard)
        elif args['command'] == 'NEW':
            if args['mode'] == 'AI':
                update.effective_message.edit_text('Игра найдена')
                new = getattr(boardgame_api, args['game']).AIMatch(update.effective_user, update.effective_chat, bot = context.bot)
                context.bot_data['matches'][new.id] = new
                new.init_turn(setup = True)
                
            elif len(context.bot_data['queue'][args['game']]) > 0:
                queued_user, queued_chat, queued_msg = context.bot_data['queue'][args['game']].pop(0)
                if queued_chat == update.effective_chat:
                    new = getattr(boardgame_api, args['game']).GroupMatch(queued_user,
                                            update.effective_user,
                                            update.effective_chat.id,
                                            bot = context.bot)
                else:
                    new = getattr(boardgame_api, args['game']).PMMatch(queued_user,
                                            update.effective_user,
                                            queued_chat.id,
                                            update.effective_chat.id,
                                            bot = context.bot)
                queued_msg.edit_text(text = 'Игра найдена')
                update.effective_message.edit_text(text = 'Игра найдена')
                context.bot_data['matches'][new.id] = new
                new.init_turn()
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
        res = res if res else (None, False)
        if context.bot_data['matches'][args['target_id']].finished:
            del context.bot_data['matches'][args['target_id']]
        update.callback_query.answer(text = res[0], show_alert = res[1])
updater.dispatcher.add_handler(tg_ext.CallbackQueryHandler(button_callback))
            
updater.bot.send_message(chat_id=os.environ['CREATOR_ID'], text='Бот включен')
updater.start_polling(drop_pending_updates=True)
updater.idle()
