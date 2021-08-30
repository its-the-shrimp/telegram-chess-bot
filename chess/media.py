import cv2
import PIL.Image
import numpy
import os

MOVETYPE_COLORS = {
    "normal": "#00cc36",
    "killing": "cc0000",
    "castling": "#3ba7ff",
    "promotion": "#3ba7ff",
    "killing-promotion": "#3ba7ff",
}


def from_hex(src):
    src = src.strip("#")
    res = []
    for band in [src[2 * i : 2 * i + 2] for i in range(3)]:
        res.append(int(band, 16))

    return res


BOARD = cv2.imread("images/chess/board.png", cv2.IMREAD_UNCHANGED)
BOARD_OFFSET = (16, 100)
TILE_SIZE = 60

PIECES = {}
for name in ["Pawn", "King", "Bishop", "Rook", "Queen", "Knight"]:
    PIECES[name] = [
        cv2.imread(f"images/chess/{color}_{name.lower()}.png", cv2.IMREAD_UNCHANGED)
        for color in ["black", "white"]
    ]
POINTER = numpy.zeros((50, 50, 4), dtype=numpy.uint8)
width = POINTER.shape[0] // 2
for row in range(POINTER.shape[0]):
    for column in range(POINTER.shape[1]):
        if abs(row - width) + abs(column - width) > round(width * 1.5):
            POINTER[row][column] = from_hex(MOVETYPE_COLORS["normal"]) + [255]
        elif abs(row - width) + abs(column - width) == round(width * 1.5):
            POINTER[row][column] = from_hex(MOVETYPE_COLORS["normal"]) + [127]
INCOMING_POINTER = numpy.zeros((50, 50, 4), dtype=numpy.uint8)
for row in range(INCOMING_POINTER.shape[0]):
    for column in range(INCOMING_POINTER.shape[1]):
        if abs(row - width) + abs(column - width) < round(width * 0.5):
            INCOMING_POINTER[row][column] = from_hex(MOVETYPE_COLORS["normal"]) + [255]
        elif abs(row - width) + abs(column - width) == round(width * 0.5):
            INCOMING_POINTER[row][column] = from_hex(MOVETYPE_COLORS["normal"]) + [127]


def _paste(src, dst, topleft):
    height, width, _ = src.shape
    alpha_mod = src[:, :, 3] / 255

    for color in range(3):
        dst_part = dst[
            topleft[1] : topleft[1] + height, topleft[0] : topleft[0] + width, color
        ]
        obj = alpha_mod * src[:, :, color] + (1 - alpha_mod) * dst_part
        dst[
            topleft[1] : topleft[1] + height, topleft[0] : topleft[0] + width, color
        ] = obj


def _fill(color, dst, topleft=(0, 0), area=None, mask=1):
    if hasattr(mask, "__array_interface__"):
        alpha_mod = mask[:, :, 3] / 255
        area = (mask.shape[1], mask.shape[0]) if not area else area
    else:
        alpha_mod = mask
    color[2], color[0] = color[0], color[2]
    for band in range(3):
        dst[
            topleft[1] : topleft[1] + area[1], topleft[0] : topleft[0] + area[0], band
        ] = (
            alpha_mod * color[band]
            + (1 - alpha_mod)
            * dst[
                topleft[1] : topleft[1] + area[1],
                topleft[0] : topleft[0] + area[0],
                band,
            ]
        )


def _imagepos(pos, size):
    return [
        BOARD_OFFSET[0] + TILE_SIZE // 2 + TILE_SIZE * pos.column - size[0] // 2,
        BOARD.shape[0]
        - BOARD_OFFSET[1]
        - TILE_SIZE // 2
        - TILE_SIZE * pos.row
        - size[1] // 2,
    ]


def _board_image(
    board,
    selected=None,
    possible_moves=[],
    prev_move=None,
):
    board_img = BOARD.copy()

    for piece in board.board:
        piece_image = PIECES[type(piece).__name__][int(piece.is_white)]
        if piece.pos == selected:
            _fill(
                from_hex("#00cc36"),
                board_img,
                topleft=_imagepos(piece.pos, piece_image.shape[:2]),
                mask=piece_image,
            )
        else:
            _paste(
                piece_image,
                board_img,
                _imagepos(piece.pos, piece_image.shape[:2]),
            )

        if type(piece).__name__ == "King" and piece.in_check():
            _fill(
                from_hex(MOVETYPE_COLORS["killing"]),
                board_img,
                topleft=_imagepos(piece.pos, INCOMING_POINTER.shape[:2]),
                mask=INCOMING_POINTER,
            )

    for new_move in possible_moves:
        _fill(
            from_hex(MOVETYPE_COLORS[new_move.type]),
            board_img,
            topleft=_imagepos(new_move.dst, POINTER.shape[:2]),
            mask=POINTER,
        )

    if prev_move:
        for move in [prev_move.src, prev_move.dst]:
            _fill(
                from_hex(MOVETYPE_COLORS["normal"]),
                board_img,
                topleft=_imagepos(move, INCOMING_POINTER.shape[:2]),
                mask=INCOMING_POINTER,
            )

    return board_img


def board_image(*args, **kwargs):
    return cv2.imencode(".jpg", _board_image(*args, **kwargs))[1].tobytes()


def board_video(match):
    path = os.path.join("images", "temp", match.video_filename)
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, BOARD.shape[1::-1]
    )

    for board, move in zip(match.states, match.get_moves()):
        img_array = _board_image(board, prev_move=move)
        for i in range(15):
            writer.write(img_array)
    for i in range(15):
        writer.write(img_array)

    thumbnail = cv2.resize(img_array, None, fx=0.5, fy=0.5)
    writer.release()
    video_data = open(path, "rb").read()
    os.remove(path)
    return video_data, cv2.imencode(".jpg", thumbnail)[1].tobytes()
