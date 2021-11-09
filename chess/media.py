import math
from typing import Union
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy
import os
import colorsys
from .core import BoardInfo, Move, MoveEval
from .parsers import PGNParser, get_moves
from .utils import BoardPoint
from .base import create_match_id, langtable, get_tempfile_url
from .analysis import eval_pieces_defense, ChessEngine, EvalScore


MOVETYPE_COLORS = {
    "normal": "#00cc36",
    "killing": "#cc0000",
    "castling": "#3ba7ff",
    "promotion": "#3ba7ff",
    "killing-promotion": "#3ba7ff",
}
EVALUATION_COLORS = {
    MoveEval.PRECISE: "#12bddb",
    MoveEval.BEST: "#3ac25a",
    MoveEval.GREAT: "#3ac25a",
    MoveEval.GOOD: "#8bcf48",
    MoveEval.WEAK: "#e0e058",
    MoveEval.MISTAKE: "#d4904c",
    MoveEval.BLUNDER: "#d1423d",
    MoveEval.FORCED: "#9c9c9c",
}
MOVE_EVAL_DESC = {k: v["move-eval-desc"] for k, v in langtable.items()}

LARGE_FONT = ImageFont.truetype("Arial-unicode.ttf", 24)
SMALL_FONT = LARGE_FONT.font_variant(size=20)


BOARD = Image.open("images/static/board.png")
BOARD_OFFSET = (16, 100)
TILE_SIZE = 60

PIECES = {}
THUMBS = {}
for name in ["Pawn", "King", "Bishop", "Rook", "Queen", "Knight"]:
    PIECES[name] = [
        Image.open(f"images/static/{color}_{name.lower()}.png")
        for color in ["black", "white"]
    ]
    THUMBS[name] = [piece_img.resize((24, 24)) for piece_img in PIECES[name]]

POINTER = numpy.zeros((50, 50, 4), dtype=numpy.uint8)
width = POINTER.shape[0] // 2
for row in range(POINTER.shape[0]):
    for column in range(POINTER.shape[1]):
        if abs(row - width) + abs(column - width) > round(width * 1.5):
            POINTER[row][column] = [255, 255, 255, 255]
        elif abs(row - width) + abs(column - width) == round(width * 1.5):
            POINTER[row][column] = [255, 255, 255, 127]
POINTER = Image.fromarray(POINTER)

INCOMING_POINTER = numpy.zeros((50, 50, 4), dtype=numpy.uint8)
for row in range(INCOMING_POINTER.shape[0]):
    for column in range(INCOMING_POINTER.shape[1]):
        if abs(row - width) + abs(column - width) < round(width * 0.5):
            INCOMING_POINTER[row][column] = [255, 255, 255, 255]
        elif abs(row - width) + abs(column - width) == round(width * 0.5):
            INCOMING_POINTER[row][column] = [255, 255, 255, 127]
INCOMING_POINTER = Image.fromarray(INCOMING_POINTER)


def _imagepos(pos: BoardPoint, size: Union[list, tuple, set]):
    return [
        BOARD_OFFSET[0] + TILE_SIZE // 2 + TILE_SIZE * pos.file - size[0] // 2,
        BOARD.height
        - BOARD_OFFSET[1]
        - TILE_SIZE // 2
        - TILE_SIZE * pos.rank
        - size[1] // 2,
    ]


def _board_image(
    lang_code: str,
    boards: list["BoardInfo"],
    selected: "BoardPoint" = None,
    possible_moves: list["Move"] = [],
    player1_name: str = "",
    player2_name: str = "",
    move_evaluation: MoveEval = None,
    pos_evaluation: EvalScore = None,
    best_move: Move = None,
    best_move_eval: EvalScore = None,
    custom_bg_pic: Image.Image = None,
):
    board_img = (custom_bg_pic or BOARD).copy()
    editor = ImageDraw.Draw(board_img)
    if not lang_code:
        lang_code = "en"

    for piece in boards[-1].board:
        piece_image = PIECES[type(piece).__name__][int(piece.is_white)]
        if piece.pos == selected:
            board_img.paste(
                MOVETYPE_COLORS["normal"],
                box=_imagepos(piece.pos, piece_image.size),
                mask=piece_image,
            )
        else:
            board_img.paste(
                piece_image,
                box=_imagepos(piece.pos, piece_image.size),
                mask=piece_image,
            )

        if type(piece).__name__ == "King" and piece.in_check():
            board_img.paste(
                MOVETYPE_COLORS["killing"],
                box=_imagepos(piece.pos, INCOMING_POINTER.size),
                mask=INCOMING_POINTER,
            )

    for new_move in possible_moves:
        board_img.paste(
            MOVETYPE_COLORS[new_move.type],
            box=_imagepos(new_move.dst, POINTER.size),
            mask=POINTER,
        )

    prev_moves = list(get_moves(boards))
    if prev_moves:
        for move in [prev_moves[-1].src, prev_moves[-1].dst]:
            board_img.paste(
                MOVETYPE_COLORS["normal"],
                box=_imagepos(move, INCOMING_POINTER.size),
                mask=INCOMING_POINTER,
            )

    offset_mult = THUMBS["Pawn"][1].width // 2
    for is_white, y in ((True, 50), (False, 606)):
        offset = 0
        for piece_type, count in boards[-1].get_taken_pieces(is_white).items():
            for i in range(count):
                board_img.paste(
                    THUMBS[piece_type.__name__][int(is_white)],
                    box=(472 - int(offset * offset_mult), y),
                    mask=THUMBS[piece_type.__name__][int(is_white)],
                )
                offset += 1
            if count:
                offset += 1.5

    editor.text((16, 664), player1_name, fiil="white", font=LARGE_FONT, anchor="ld")
    editor.text((16, 16), player2_name, fill="white", font=LARGE_FONT)

    blacks_value = sum([piece.value for piece in boards[-1].blacks])
    whites_value = sum([piece.value for piece in boards[-1].whites])
    if blacks_value != whites_value:
        editor.text(
            (496, 664 if whites_value > blacks_value else 16),
            f"+{abs(whites_value - blacks_value)}",
            fill="white",
            font=SMALL_FONT,
            anchor="rd" if whites_value > blacks_value else "ra",
        )

    lines = PGNParser.encode_moveseq(
        moves=prev_moves, result=None, language_code="emoji", turns_per_line=1
    ).splitlines()
    max_length = max([SMALL_FONT.getlength(line) for line in lines] + [168])
    space_length = round(SMALL_FONT.getlength(" "))
    y_offset = 0
    for line in reversed(lines[-5:]):
        cur_line_length = SMALL_FONT.getlength(line)
        tokens = line.split(" ")
        if not y_offset and move_evaluation:
            if len(tokens) == 2:
                editor.rectangle(
                    (522, 295, 532 + cur_line_length, 325),
                    fill=EVALUATION_COLORS[move_evaluation],
                )
            elif len(tokens) == 3:
                editor.rectangle(
                    (700 - SMALL_FONT.getlength(tokens[2]), 295, 710, 325),
                    fill=EVALUATION_COLORS[move_evaluation],
                )
            editor.rectangle(
                (522, 325, 710, 355), fill=EVALUATION_COLORS[move_evaluation]
            )
            editor.text(
                (537, 340),
                move_evaluation.value[:2],
                fill="white",
                font=LARGE_FONT,
                anchor="mm",
            )
            editor.text(
                (552, 340),
                MOVE_EVAL_DESC[lang_code][move_evaluation.value],
                fill="white",
                font=SMALL_FONT,
                anchor="lm",
            )

        padded = tokens[0] + " " + tokens[1]
        if len(tokens) == 3:
            padded += (
                " " * round((max_length - cur_line_length) // space_length + 1)
                + tokens[2]
            )
        editor.text(
            (532, BOARD.height // 2 - y_offset - 15),
            padded,
            fill="white",
            font=SMALL_FONT,
            anchor="ld",
        )
        y_offset += round(SMALL_FONT.size * 1.5)

    bar_x_offset = None
    if pos_evaluation is not None:
        if pos_evaluation.mate_in != 0 and pos_evaluation.score == 0:
            bar_x_offset = -94 if pos_evaluation.mate_in < 0 else 94
            editor.rectangle(
                (522, 385, 710, 415),
                fill="#233139" if pos_evaluation.mate_in < 0 else "white",
            )
            editor.text(
                (616, 400),
                str(pos_evaluation),
                font=SMALL_FONT,
                fill="white" if pos_evaluation.mate_in < 0 else "#233139",
                anchor="mm",
            )
        else:
            formatted = str(pos_evaluation)
            editor.text(
                (616, 400), formatted, fill="white", anchor="mm", font=SMALL_FONT
            )
            bar_x_offset = round(math.log(abs(pos_evaluation.score) + 1, 10) * 47) * (
                1 if pos_evaluation.score > 0 else -1
            )
            eval_bar = Image.new("RGB", (94 + bar_x_offset, 30), color="white")
            ImageDraw.Draw(eval_bar).text(
                (94, 15), formatted, font=SMALL_FONT, fill="#233139", anchor="mm"
            )
            board_img.paste(eval_bar, (522, 385))

    if best_move is not None and move_evaluation not in [MoveEval.BEST, MoveEval.PRECISE, MoveEval.FORCED]:
        editor.rectangle(
            (522, 415, 710, 475),
            fill=EVALUATION_COLORS[MoveEval.BEST],
        )
        editor.text(
            (616, 430),
            langtable[lang_code]["best-move"],
            fill="white",
            anchor="mm",
            font=LARGE_FONT
        )
        editor.text(
            (527, 460), 
            best_move.pgn_encode(language_code="emoji"),
            fill="white",
            anchor="lm",
            font=LARGE_FONT
        )
        if best_move_eval is not None:
            editor.text(
                (705, 460),
                str(best_move_eval),
                fill="white",
                anchor="rm",
                font=LARGE_FONT
            )

    array = numpy.array(board_img)
    temp = (array[:, :, 0].copy(), array[:, :, 2].copy())
    array[:, :, 2], array[:, :, 0] = temp
    return array


def _board_image_with_static_analysis(boards: list[BoardInfo], **kwargs):
    board_img = BOARD.copy()
    editor = ImageDraw.Draw(board_img, mode="RGBA")
    eval_res = eval_pieces_defense(boards[-1])
    for pos, score in eval_res.items():
        topleft = _imagepos(pos, (60, 60))
        if score == 0:
            h = 0.1667
        elif score > 0:
            h = 0.33
        elif score < 0:
            h = 0.0
        editor.rectangle(
            (*topleft, topleft[0] + 60, topleft[1] + 60),
            fill=tuple(
                [
                    round(i * 255)
                    for i in colorsys.hls_to_rgb(h, 0.75, 0.7 + abs(score) * 10)
                ]
            ),
        )

    return _board_image(boards, custom_bg_pic=board_img, **kwargs)


def board_image(*args, **kwargs) -> bytes:
    return cv2.imencode(".jpg", _board_image(*args, **kwargs))[1].tobytes()


def board_image_with_static_analysis(*args, **kwargs) -> bytes:
    return cv2.imencode(".jpg", _board_image_with_static_analysis(*args, **kwargs))[
        1
    ].tobytes()


def board_video(
    match,
    lang_code: str,
    player1_name: str = None,
    player2_name: str = None,
    analyser: ChessEngine = None,
) -> tuple[bytes, bytes]:
    path = os.path.join("images", "temp", create_match_id(n=16) + ".mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, BOARD.size)
    for index in range(len(match.boards)):
        move = match.boards[index] - match.boards[index - 1] if index else None
        prev_move = (
            match.boards[index - 1] - match.boards[index - 2] if index > 1 else None
        )
        move_eval, best_move, best_move_eval = (
            analyser.eval_move(move, prev_move=prev_move)
            if index and analyser
            else (None, None, None)
        )

        img_array = _board_image(
            lang_code,
            match.boards[: index + 1],
            player1_name=player1_name or match.db.get_name(match.player1),
            player2_name=player2_name or match.db.get_name(match.player2),
            move_evaluation=move_eval,
            best_move=best_move,
            pos_evaluation=analyser.eval_position(move) if move and analyser else None,
            best_move_eval=best_move_eval
        )

        for i in range(15):
            writer.write(img_array)
    for i in range(15):
        writer.write(img_array)
    writer.release()

    thumbnail = cv2.resize(img_array, (200, 200))
    video_data = open(path, "rb").read()
    os.remove(path)

    return video_data, cv2.imencode(".jpg", thumbnail)[1].tobytes()
