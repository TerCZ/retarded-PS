import cv2
import numpy as np
import threading
import multiprocessing

from enum import Enum
from math import ceil

from .util import *


class Channel(Enum):
    Blue = 0
    Green = 1
    Red = 2


def read(path, greyscale=False):
    """
    Use openCV to read image. Assume color image by default.
    :param path: path to image
    :param greyscale: set to True to convert to greyscale image
    :return: numpy array of image, in BGR or greyscale mode
    """
    if greyscale:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)

    if img is None:
        logging.error("cannot read {}".format(path))
        return None

    img = img.astype(np.float64)
    img /= 255

    return img


def write(path, img_original):
    """
    Use openCV to write image.
    :param path: path to save image
    :param img_original: numpy array of image, in BGR or greyscale mode
    :return: return value from cv2.imwrite
    """
    # img_original = img_original.copy()
    img = img_original.copy()
    img *= 255
    return cv2.imwrite(path, img)


def get_channel(img, channel):
    """
    Extract color chanel of image.
    :param img: numpy array of color image, in BGR mode
    :param channel: index of chanel (only 0, 1, 2 are valid)
    :return: numpy array of target channel, share data with input img
    """
    if is_greyscale(img):
        logger.warning("cannot extract color channel from greyscale image")
        return img

    if channel == Channel.Red:
        channel = 2
    elif channel == Channel.Green:
        channel = 1
    elif channel == Channel.Blue:
        channel = 0

    return img[:, :, channel]


def bgr_to_grey(img):
    """
    Convert color image to greyscale, using formula Y = 0.299 * R + 0.587 * G + 0.114 * B
    :param img: numpy array of color image, in BGR mode
    :return: numpy array of greyscale image
    """
    if is_greyscale(img):
        logger.info("bgr_to_greyscale: try to convert greyscale image to greyscale")
        return img

    weight = np.array([0.114, 0.587, 0.299]).reshape((3, 1))
    img = img.dot(weight).reshape(img.shape[:2])
    return img


def grey_to_rgb(img):
    img = img.reshape((*img.shape, 1))
    return np.concatenate((img, img, img), axis=2)


def bgr_to_rgb(img):
    if is_greyscale(img):
        logger.info("bgr_to_rgb: try to convert greyscale image to hsv")
        return img

    result = img.copy()
    result[:, :, 0] = img[:, :, 2]
    result[:, :, 2] = img[:, :, 0]

    return result


def bgr_to_hsv(img):
    """
    Convert color image from BGR mode to HSV mode. H in [0, 360), S in [0, 1], V in [0, 1]
    :param img: numpy array of color image, in BGR mode
    :return: numpy array of color image, in HSV mode
    """
    if is_greyscale(img):
        logger.info("try to convert greyscale image to hsv")
        return img

    def transform(pixel):
        b, g, r = pixel
        max_val, min_val = max(b, g, r), min(b, g, r)

        # hue
        if max_val == min_val:
            h = 0
        elif max_val == r:
            h = 60 * (g - b) / (max_val - min_val)
            if g < b:
                h += 360
        elif max_val == g:
            h = 60 * (b - r) / (max_val - min_val) + 120
        else:
            h = 60 * (r - g) / (max_val - min_val) + 240

        # saturation
        if max_val == 0:
            s = 0
        else:
            s = 1 - min_val / max_val

        # value
        v = max_val

        return h, s, v

    return iter_pixel(img, transform)


def hsv_to_bgr(img):
    """
    Convert color image from HSV mode to BGR mode. H in [0, 360), S in [0, 1], V in [0, 1]
    :param img: numpy array of color image, in HSV mode
    :return: numpy array of color image, in BGR mode
    """
    if is_greyscale(img):
        logger.info("try to convert greyscale image to bgr")
        return img

    def transform(pixel):
        h, s, v = pixel

        h_i = h // 60
        f = h / 60 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        if h_i == 0:
            return p, t, v
        elif h_i == 1:
            return p, v, q
        elif h_i == 2:
            return t, v, p
        elif h_i == 3:
            return v, q, p
        elif h_i == 4:
            return v, p, t
        else:
            return q, p, v

    return iter_pixel(img, transform)


def bgr_to_hsl(img):
    """
    Convert color image from BGR mode to HSL mode. H in [0, 360), S in [0, 1], L in [0, 1]
    :param img: numpy array of color image, in BGR mode
    :return: numpy array of color image, in HSL mode
    """
    if is_greyscale(img):
        logger.info("try to convert greyscale image to hsl")
        return img

    def transform(pixel):
        b, g, r = pixel
        max_val, min_val = max(b, g, r), min(b, g, r)

        # hue
        if max_val == min_val:
            h = 0
        elif max_val == r:
            h = 60 * (g - b) / (max_val - min_val)
            if g < b:
                h += 360
        elif max_val == g:
            h = 60 * (b - r) / (max_val - min_val) + 120
        else:
            h = 60 * (r - g) / (max_val - min_val) + 240

        # lightness
        l = (max_val + min_val) / 2

        # saturation
        if l == 0 or max_val == min_val:
            s = 0
        elif l <= 0.5:
            s = (max_val - min_val) / 2 / l
        else:
            s = (max_val - min_val) / (2 - 2 * l)

        return h, s, l

    return iter_pixel(img, transform)


def hsl_to_bgr(img):
    """
    Convert color image from HSL mode to BGR mode. H in [0, 360), S in [0, 1], L in [0, 1]
    :param img: numpy array of color image, in HSL mode
    :return: numpy array of color image, in BGR mode
    """
    if is_greyscale(img):
        logger.info("try to convert greyscale image to bgr")
        return img

    def transform(pixel):
        h, s, l = pixel

        c = (1 - abs(2 * l - 1)) * s
        h_i = h / 60
        x = c * (1 - abs(h_i % 2 - 1))
        m = l - 0.5 * c

        if h_i <= 1:
            return m, x + m, c + m
        elif h_i <= 2:
            return m, c + m, x + m
        elif h_i <= 3:
            return x + m, c + m, m
        elif h_i <= 4:
            return c + m, x + m, m
        elif h_i <= 5:
            return c + m, m, x + m
        elif h_i <= 6:
            return x + m, m, c + m

    return iter_pixel(img, transform)


def adjust_hue(img, offset):
    """
    Increment hue value by offset, divide incremented value by 360, and use the remainder as new hue.
    This function will modify input img.
    :param img: numpy array of color image in HSV/HSL mode, H in [0, 360), S in [0, 1], V in [0, 1]
    :param offset: integer in [-180, 180]
    :return: numpy array of adjusted color image in HSV/HSL mode
    """
    if is_greyscale(img):
        logger.info("try to adjust hue of greyscale image")
        return img

    img = img.copy()
    img[:, :, 0] = (img[:, :, 0] + offset) % 360
    return img


def adjust_saturation(img, factor):
    """
    Multiply saturation value by factor, divide multiplied value by 1, and use the remainder as new saturation.
    This function will modify input img.
    :param img: numpy array of color image in HSV/HSL mode, H in [0, 360), S in [0, 1], V in [0, 1]
    :param factor: positive float
    :return: numpy array of adjusted color image in HSV/HSL mode
    """
    if is_greyscale(img):
        logger.info("try to adjust saturation of greyscale image")
        return img

    img = img.copy()
    img[:, :, 1] = img[:, :, 1] * factor
    img[img[:, :, 1] > 1, 1] = 1
    return img


def adjust_value(img, factor):
    """
    Multiply value (of HSV) value by factor, divide multiplied value by 1, and use the remainder as new saturation.
    This function will modify input img.
    :param img: numpy array of color image in HSV mode, H in [0, 360), S in [0, 1], V in [0, 1]
    :param factor: positive float
    :return: numpy array of adjusted color image in HSV mode
    """
    if is_greyscale(img):
        logger.info("try to adjust value (of HSV) of greyscale image")
        return img

    img = img.copy()
    img[:, :, 2] = img[:, :, 2] * factor
    img[img[:, :, 2] > 1, 2] = 1
    return img


def adjust_lightness(img, factor):
    """
    Multiply saturation value by factor, divide multiplied value by 1, and use the remainder as new saturation.
    This function will modify input img.
    :param img: numpy array of color image in HSL mode, H in [0, 360), S in [0, 1], L in [0, 1]
    :param factor: positive float
    :return: numpy array of adjusted color image in HSL mode
    """
    if is_greyscale(img):
        logger.info("try to adjust lightness of greyscale image")
        return img

    img = img.copy()
    img[:, :, 2] = img[:, :, 2] * factor
    img[img[:, :, 2] > 1, 2] = 1
    return img


def rgb2hsl(rgb):
    def core(_rgb, _hsl):

        irgb = _rgb.astype(np.uint16)
        ir, ig, ib = irgb[:, :, 0], irgb[:, :, 1], irgb[:, :, 2]
        h, s, l = _hsl[:, :, 0], _hsl[:, :, 1], _hsl[:, :, 2]

        imin, imax = irgb.min(2), irgb.max(2)
        iadd, isub = imax + imin, imax - imin

        ltop = (iadd != 510) * (iadd > 255)
        lbot = (iadd != 0) * (ltop == False)

        l[:] = iadd.astype(np.float) / 510

        fsub = isub.astype(np.float)
        s[ltop] = fsub[ltop] / (510 - iadd[ltop])
        s[lbot] = fsub[lbot] / iadd[lbot]

        not_same = imax != imin
        is_b_max = not_same * (imax == ib)
        not_same_not_b_max = not_same * (is_b_max == False)
        is_g_max = not_same_not_b_max * (imax == ig)
        is_r_max = not_same_not_b_max * (is_g_max == False) * (imax == ir)

        h[is_r_max] = ((0. + ig[is_r_max] - ib[is_r_max]) / isub[is_r_max])
        h[is_g_max] = ((0. + ib[is_g_max] - ir[is_g_max]) / isub[is_g_max]) + 2
        h[is_b_max] = ((0. + ir[is_b_max] - ig[is_b_max]) / isub[is_b_max]) + 4
        h[h < 0] += 6
        h[:] /= 6

    hsl = np.zeros(rgb.shape, dtype=np.float)
    cpus = multiprocessing.cpu_count()
    length = int(ceil(float(hsl.shape[0]) / cpus))
    line = 0
    threads = []
    while line < hsl.shape[0]:
        line_next = line + length
        thread = threading.Thread(target=core, args=(rgb[line:line_next], hsl[line:line_next]))
        thread.start()
        threads.append(thread)
        line = line_next

    for thread in threads:
        thread.join()

    return hsl


def hsl2rgb(hsl):
    def core(_hsl, _frgb):

        h, s, l = _hsl[:, :, 0], _hsl[:, :, 1], _hsl[:, :, 2]
        fr, fg, fb = _frgb[:, :, 0], _frgb[:, :, 1], _frgb[:, :, 2]

        q = np.zeros(l.shape, dtype=np.float)

        lbot = l < 0.5
        q[lbot] = l[lbot] * (1 + s[lbot])

        ltop = lbot == False
        l_ltop, s_ltop = l[ltop], s[ltop]
        q[ltop] = (l_ltop + s_ltop) - (l_ltop * s_ltop)

        p = 2 * l - q
        q_sub_p = q - p

        is_s_zero = s == 0
        l_is_s_zero = l[is_s_zero]
        per_3 = 1. / 3
        per_6 = 1. / 6
        two_per_3 = 2. / 3

        def calc_channel(channel, t):
            t[t < 0] += 1
            t[t > 1] -= 1
            t_lt_per_6 = t < per_6
            t_lt_half = (t_lt_per_6 == False) * (t < 0.5)
            t_lt_two_per_3 = (t_lt_half == False) * (t < two_per_3)
            t_mul_6 = t * 6

            channel[:] = p.copy()
            channel[t_lt_two_per_3] = p[t_lt_two_per_3] + q_sub_p[t_lt_two_per_3] * (4 - t_mul_6[t_lt_two_per_3])
            channel[t_lt_half] = q[t_lt_half].copy()
            channel[t_lt_per_6] = p[t_lt_per_6] + q_sub_p[t_lt_per_6] * t_mul_6[t_lt_per_6]
            channel[is_s_zero] = l_is_s_zero.copy()

        calc_channel(fr, h + per_3)
        calc_channel(fg, h.copy())
        calc_channel(fb, h - per_3)

    frgb = np.zeros(hsl.shape, dtype=np.float)
    cpus = multiprocessing.cpu_count()
    length = int(ceil(float(hsl.shape[0]) / cpus))
    line = 0
    threads = []
    while line < hsl.shape[0]:
        line_next = line + length
        thread = threading.Thread(target=core, args=(hsl[line:line_next], frgb[line:line_next]))
        thread.start()
        threads.append(thread)
        line = line_next

    for thread in threads:
        thread.join()

    return (frgb * 255).round().astype(np.uint8)
