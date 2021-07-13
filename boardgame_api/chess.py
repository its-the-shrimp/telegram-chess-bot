import PIL.Image
import itertools
import string
import random
import io

IDSAMPLE = string.ascii_uppercase + string.ascii_lowercase + string.digits + '-+'

def _imgpos(pos):
    return [19 + 60 * pos[0] - pos[0] // 2, 422 - 59 * pos[1]]

def decode_pos(pos):
    return (ord(pos[0]) - 65, int(pos[1]) - 1)

def encode_pos(pos):
    return chr(pos[0] + 65) + str(pos[1] + 1)

def in_bounds(pos):
    return 0 <= pos[0] <= 7 and 0 <= pos[1] <= 7

class BaseFigure():
    name = 'PLACEHOLDER'
    
    @staticmethod
    def _getmoves(pos, match, is_white):
        return []
    
    @classmethod
    def _checkmate(cls, pos, match, is_white):
        enemies = match.blacks if is_white else match.whites
        
        enemy_moves = itertools.chain(*[i.get_moves() for i in enemies])
        enemy_moves = [i['pos'] for i in enemy_moves]
        self_positions = [pos] + [i['pos'] for i in cls._getmoves(pos, match, is_white)]
        return all([possible_pos in enemy_moves for possible_pos in self_positions])
    
    def __init__(self, pos, match, is_white):
        try:
            self.image = PIL.Image.open(f"images/chess/{'white' if is_white else 'black'}_{type(self).__name__}.png")
        except:
            self.image = PIL.Image.new('RGB', (57, 56), color = '#ffffff' if is_white else '#000000')
        self.pos = pos
        self.is_white = is_white
        self.match = match
        
    def __str__(self):
        return f"{self.name} на {encode_pos(self.pos)}"
        
    def get_moves(self):
        return self._getmoves(self.pos, self.match, self.is_white)
    
    def move(self, pos):
        if pos == self.match.passing_pos[1]:
            killed = self.match[self.match.passing_pos[0]]
            del self.match[self.match.passing_pos[0]]
            return killed
        
        figure = self.match[pos]
        if figure:
            del self.match[pos]
            return figure
        
        self.pos = pos
        self.passing_pos = [None, None]
    
    def in_checkmate(self):
        return self._checkmate(self.pos, self.match, self.is_white)
    
class Pawn(BaseFigure):
    name = 'Пешка'
    @staticmethod
    def _getmoves(pos, match, is_white):
        allies, enemies = (match.whites, match.blacks) if is_white else (match.blacks, match.whites)
        
        if [pos[0], pos[1] + (1 if is_white else -1)] not in [i.pos for i in enemies]:
            positions = [[pos[0], pos[1] + (1 if is_white else -1)]]
        if ((pos[1] == 1 and is_white) or (pos[1] == 6 and not is_white)) and [pos[0], pos[1] + (2 if is_white else -2)] not in [i.pos for i in enemies]:
            positions.append([pos[0], pos[1] + (2 if is_white else -2)])
            
        if [pos[0] + 1, pos[1] + 1] in [i.pos for i in enemies]:
            positions.append([pos[0] + 1, pos[1] + 1])
            
        elif [pos[0] + 1, pos[1] - 1] in [i.pos for i in enemies]:
            positions.append([pos[0] + 1, pos[1] - 1])
        
        moves = []
        for move in positions:
            if in_bounds(move) and move not in [i.pos for i in allies]:
                moves.append({'pos': move, 'killing': move in [i.pos for i in enemies]})
                
        return moves
    
    def move(self, pos):
        old_pos = self.pos
        super().move(pos)
        if abs(old_pos[1] - pos[1]) == 2:
            self.match.passing_pos = [pos, [pos[0], (pos[1] + old_pos[1]) // 2]]
                
class Knight(BaseFigure):
    name = 'Конь'
    @staticmethod
    def _getmoves(pos, match, is_white):
        allies, enemies = (match.whites, match.blacks) if is_white else (match.blacks, match.whites)
        moves = []
        for move in [[2, -1], [2, 1], [1, 2], [1, -2], [-1, 2], [-1, -2], [-2, 1], [-2, -1]]:
            move = [a+b for a,b in zip(move, pos)]
            if in_bounds(move) and move not in [i.pos for i in allies]:
                moves.append({'pos': move, 'killing': move in [i.pos for i in enemies]})
                
        return moves
    
class Rook(BaseFigure):
    name = 'Ладья'
    @staticmethod
    def _getmoves(pos, match, is_white):
        allies, enemies = (match.whites, match.blacks) if is_white else (match.blacks, match.whites)
        moves = []
        for move_seq in [zip([pos[0]] * (7 - pos[0]), range(pos[0] + 1, 8)), zip([pos[0]] * pos[0], range(pos[0])),
                         zip(range(pos[0] + 1, 8), [pos[0]] * (7 - pos[0])), zip(range(pos[0]), [pos[0]] * pos[0])]:
            for move in move_seq:
                if move in [i.pos for i in allies]:
                    break
                elif move in [i.pos for i in enemies]:
                    moves.append({'pos': move, 'killing': True})
                    break
                else:
                    moves.append({'pos': move, 'killing': False})
                    
        return moves
    
class Bishop(BaseFigure):
    name = 'Слон'
    @staticmethod
    def _getmoves(pos, match, is_white):
        allies, enemies = (match.whites, match.blacks) if is_white else (match.blacks, match.whites)
        moves = []
        for move_seq in [zip(range(pos[0] + 1, 8), range(pos[0] + 1, 8)), zip(range(pos[0] + 1, 8), range(pos[0])),
                         zip(range(pos[0]), range(pos[0] + 1, 8)), zip(range(pos[0]), range(pos[0]))]:
            for move in move_seq:
                if move in [i.pos for i in allies]:
                    break
                elif move in [i.pos for i in enemies]:
                    moves.append({'pos': move, 'killing': True})
                    break
                else:
                    moves.append({'pos': move, 'killing': False})
                    
        return moves
    
class Queen(BaseFigure):
    name = 'Ферзь'
    @staticmethod
    def _getmoves(pos, match, is_white):
        return Bishop._getmoves(pos, match, is_white) + Rook._getmoves(pos, match, is_white)
    
class King(BaseFigure):
    name = 'Король'
    @staticmethod
    def _getmoves(pos, match, is_white):
        allies, enemies = (match.whites, match.blacks) if is_white else (match.blacks, match.whites)
        moves = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                move = [pos[0] + x, pos[1] + y]
                if [x, y] == [0, 0] and in_bounds(move) and move not in [i.pos for i in allies]:
                    moves.append({'pos': move, 'killing': move in [i.pos for i in enemies]})
                
        return moves
    
    @staticmethod
    def _check(pos, match, is_white):
        allies, enemies = (match.whites, match.blacks) if is_white else (match.blacks, match.whites)
        enemy_moves = itertools.chain(*[i.get_moves() for i in enemies])
        
        return pos in [i['pos'] for i in enemy_moves]
    
    def in_check(self):
        return self._check(self.pos, self.match, self.is_white)
    
class BaseMatch():
    BOARD_IMG = PIL.Image.open("images/chess/board.png")
    POINTER_IMG = PIL.Image.open("images/chess/pointer.png")
    WRONG_PERSON_MSG = 'Сейчас не ваш ход!'
    
    def __init__(self):
        self.turn = 1
        self.is_white_turn = True
        self.passing_pos = None
        self.whites = []
        self.blacks = []
        self.finished = False
        self.id = ''.join(random.choices(IDSAMPLE, k = 8))
        for x in range(8):
            self.whites.append(Pawn([x, 1], self, True))
            self.blacks.append(Pawn([x, 6], self, False))
        
        for x, figure in enumerate([Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]):
            self.whites.append(figure([x, 0], self, True))
            self.blacks.append(figure([x, 7], self, False))
            
    def __getitem__(self, key):
        for figure in self.whites + self.blacks:
            if figure.pos == key:
                return figure
            
    def __setitem__(self, key, value):
        for figure in self.whites + self.blacks:
            if figure.pos == key:
                figure.pos = value
                
    def __delitem__(self, key):
        for figure in self.whites + self.blacks:
            if figure.pos == key:
                del figure
            
    def get_context(self):
        return {'allies': self.whites if self.is_white_turn else self.blacks,
                'enemies': self.blacks if self.is_white_turn else self.whites,
                'white_turn': self.is_white_turn}
    
    def get_king(self, is_white):
        for figure in (self.whites if is_white else self.blacks):
            if type(figure) == King:
                return figure
    
    def init_turn(self, move = [None, None]):
        context = self.get_context()
        figure = self[move[0]]
        turn_info  = {'figure': figure}
        if figure:
            turn_info.update({'from': encode_pos(move[0]),
                              'to': encode_pos(move[1])})
        
        if figure:
            killed = self[move[0]].move(move[1])
            turn_info.update({'killed': killed})
        else:
            turn_info.update({'killed': None})
            
        cur_king = self.get_king(not context['white_turn'])
        if cur_king.in_checkmate():
            turn_info.update({'player_gamestate': 'checkmate'})
            self.finished = True
        elif cur_king.in_check():
            turn_info.update({'player_gamestate': 'check'})
        else:
            turn_info.update({'player_gamestate': 'normal'})
        if self.is_white_turn:
            self.is_white_turn = False
        else:
            self.is_white_turn = True
            self.turn += 1
        return turn_info
            
    def assemble_board(self, selected = None, pointers = []):
        board = self.BOARD_IMG.copy()
        for figure in self.whites + self.blacks:
            board.paste('#00cc36' if figure.pos == selected else figure.image,
                        box = _imgpos(figure.pos),
                        mask = figure.image)
        
        for pointer in pointers:
            board.paste('#cc0000' if pointer['killing'] else '#00cc36',
                        box = _imgpos(pointer['pos']),
                        mask = self.POINTER_IMG)
        
        board = board.convert(mode = 'RGB')
        buffer = io.BytesIO()
        board.save(buffer, format = 'JPEG')
        return buffer.getvalue()
    
class GroupMatch(BaseMatch):
    def __init__(self, player1, player2, match_chat):
        self.player1 = player1
        self.player2 = player2
        self.chat_id = match_chat.id
        self.ids = []
        self.last_msg = {}
        super().__init__()
        
    def get_context(self):
        context = super().get_context()
        context.update({'player': self.player1 if context['white_turn'] else self.player2,
                        'opponent': self.player2 if context['white_turn'] else self.player1})
        return context
    
    def init_turn(self, move = [None, None]):
        res = super().init_turn(move = move)
        context = self.get_context()
        if res['player_gamestate'] == 'checkmate':
            msg = f"Игра окончена: шах и мат!\nХодов: {self.turn - 1}.\nПобедитель: {context['opponent'].name}."
        else:
            msg = f"Ход {self.turn}"
            if res['figure']:
                msg += f"\n{res['figure'].name}: {res['from']} -> {res['to']}"
            else:
                msg += '\n'
            
            if res['killed']:
                msg += f"\n{res['killed']} игрока {context['player'].name} убит{'а' if res['killed'].name in ['Пешка', 'Ладья'] else ''}!"
            else:
                msg += '\n'
                
            msg += f"\nХодит { context['player'].name }; выберите действие:"
            
        output = [{'msg_id': self.ids[0] if self.ids != [] else None,
                 'chat_id': self.chat_id,
                 'img': self.assemble_board(),
                 'text': msg,
                 'expected_uid': context['player'].id, 
                 'answers': [
                     [{'text': 'Ходить', 'callback_data': ['TURN']}],
                     [{'text': 'Сдаться', 'callback_data': ['SURRENDER']}]
                 ]
                }]
        
        self.last_msg = {'text': output[0]['text'], 'img': output[0]['img'], 'answers': output[0]['answers']}
        return output
    
    def handle_input(self, args):
        context = self.get_context()
        if args[0] == 'TURN':
            figure_buttons = []
            for figure in context['allies']:
                figure_buttons.append([{'text': str(figure), 'callback_data': ['CHOOSE_FIGURE', encode_pos(figure.pos)]}])
            
            new_text = self.last_msg['text'].split('\n')
            new_text[-1] = f"Ходит {context['player'].name}; выберите фигуру:"
            output = [{
                'chat_id': self.chat_id,
                'msg_id': self.ids[0],
                'text': '\n'.join(new_text),
                'img': self.last_msg['img'],
                'expected_uid': context['player'].id,
                'answers': figure_buttons
            }]
            
            self.last_msg = {'text': output[0]['text'], 'img': output[0]['img'], 'answers': output[0]['answers']}
            return output
        
        elif args[0] == 'SURRENDER':
            self.finished = True
            return [{
                'chat_id': self.chat_id,
                'msg_id': self.ids[0],
                'img': self.last_msg['img'],
                'text': f"Игра окончена: {context['player']} сдался.\nХодов: {self.turn - 1}.\nПобедитель: {context['opponent']}.",
                'answers': None
            }]
            
        
        elif args[0] == 'CHOOSE_FIGURE':
            dest_buttons = []
            moves = self[decode_pos(args[1])].get_moves()
            for move in moves:
                dest_buttons.append([{'text': ('❌' if move['killing'] else '')+encode_pos(move['pos']),
                                      'callback_data': ['MOVE', args[1], encode_pos(move['pos'])]}])
                
            new_text = self.last_msg['text'].split('\n')
            new_text[-1] = f"Ходит {context['player']}; выберите новое место фигуры:"
            output = [{
                'chat_id': self.chat_id,
                'msg_id': self.ids[0],
                'img': self.assemble_board(selected = decode_pos(args[1]), pointers = moves),
                'text': '\n'.join(new_text),
                'expected_uid': context['player'].id,
                'answers': dest_buttons
            }]
            self.last_msg = {'text': output[0]['text'], 'img': output[0]['img'], 'answers': output[0]['answers']}
            return output
        
        elif args[0] == 'MOVE':
            return self.init_turn(move = [decode_pos(args[1]), decode_pos(args[2])])
        
class PMMatch(BaseMatch):
    def __init__(self, player1, player2, chat1, chat2):
        self.player1 = player1
        self.player2 = player2
        self.chat_id1 = chat_id1.id
        self.chat_id2 = chat_id2.id
        self.ids = []
        self.last_msg = {}
        super().__init__()
        
    def init_turn(self, move = [None, None]):
        res = super().init_turn(move = move)
        context = self.get_context()
        if res['player_gamestate'] == 'checkmate':
            player_msg = opponent_msg = f"Игра окончена: шах и мат!\nХодов: {self.turn - 1}.\nПобедитель: {context['opponent'].name}."
        else:
            player_msg = f"Ход {self.turn}"
            if res['figure']:
                player_msg += f"\n{res['figure'].name}: {res['from']} -> {res['to']}"
            else:
                player_msg += '\n'
            
            if res['killed']:
                player_msg += f"\n{res['killed']} игрока {context['player'].name} убит{'а' if res['killed'].name in ['Пешка', 'Ладья'] else ''}!"
            else:
                player_msg += '\n'
                
            opponent_msg = player_msg
                
            Player_msg += '\nВыберите действие:'
            opponent_msg += f"\nХодит {context['player'].name}"
            img = self.assemble_board()
            
        output = [
            {
                'msg_id': self.ids[not context['white_turn']] if self.ids != [] else None,
                'chat_id': self.chat_id1 if context['white_turn'] else self.chat_id2,
                'img': img,
                'text': player_msg,
                'expected_uid': context['player'].id,
                'answers': [
                    [{'text': 'Ходить', 'callback_data': ['TURN']}],
                    [{'text': 'Сдаться', 'callback_data': ['SURRENDER']}]
                ]
            }, {
                'msg_id': self.ids[context['white_turn']] if self.ids != [] else None,
                'chat_id': self.chat_id2 if context['is_white_turn'] else self.chat_id1,
                'img': img,
                'text': opponent_msg
            }
        ]
        
        self.last_msg = {'player': output[0], 'opponent': output[1]}
        return output
    
    def handle_input(self, args):
        context = self.get_context()
        if args[0] == 'TURN':
            figure_buttons = []
            for figure in context['allies']:
                figure_buttons.append([{'text': str(figure), 'callback_data': ['CHOOSE_FIGURE', encode_pos(figure.pos)]}])
            
            new_text = self.last_msg['player']['text'].split('\n')
            new_text[-1] = f"Ходит {context['player'].name}; выберите фигуру:"
            output = [{
                **self.last_msg['player'],
                'text': '\n'.join(new_text),
                'answers': figure_buttons
            }]
            
            self.last_msg['player'] = output[0]
            return output
        
        elif args[0] == 'SURRENDER':
            self.finished = True
            return [{
                **self.last_msg['player'],
                'text': f"Игра окончена: {context['player']} сдался.\nХодов: {self.turn - 1}.\nПобедитель: {context['opponent']}.",
                'answers': None
            }, {
                **self.last_msg['opponent'],
                'text': f"Игра окончена: {context['player']} сдался.\nХодов: {self.turn - 1}.\nПобедитель: {context['opponent']}.",
                'answers': None
            }]
            
        
        elif args[0] == 'CHOOSE_FIGURE':
            dest_buttons = []
            moves = self[decode_pos(args[1])].get_moves()
            for move in moves:
                dest_buttons.append([{'text': ('❌' if move['killing'] else '')+encode_pos(move['pos']),
                                      'callback_data': ['MOVE', args[1], encode_pos(move['pos'])]}])
                
            new_text = self.last_msg['text'].split('\n')
            new_text[-1] = f"Ходит {context['player']}; выберите новое место фигуры:"
            output = [{
                **self.last_msg['player'],
                'img': self.assemble_board(selected = decode_pos(args[1]), pointers = moves),
                'text': '\n'.join(new_text),
                'answers': dest_buttons
            }]
            self.last_msg['player'] = output[0]
            return output
        
        elif args[0] == 'MOVE':
            return self.init_turn(move = [decode_pos(args[1]), decode_pos(args[2])])
        