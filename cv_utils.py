import cv2


def paste(src, dst, topleft):
    height, width, _ = src.shape
    alpha_mod = src[:, :, 3] / 255

    for color in range(3):
        dst[
            topleft[1] : topleft[1] + height, topleft[0] : topleft[0] + width, color
        ] = (
            alpha_mod * src[:, :, color]
            + (1 - alpha_mod)
            * dst[
                topleft[1] : topleft[1] + height, topleft[0] : topleft[0] + width, color
            ]
        )


def fill(color, dst, topleft=(0, 0), area=None, mask=1):
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


def from_hex(src):
    src = src.strip("#")
    res = []
    for band in [src[2 * i : 2 * i + 2] for i in range(3)]:
        res.append(int(band, 16))

    return res


def image_pos(pos, size):
    return [46 + 60 * pos[0] - size[0] // 2, 550 - 60 * pos[1] - size[1] // 2]
