import cv2
import numpy as np

from math import log1p, floor, ceil, sin, cos, pi
from .util import *

from .basic import *
from .contrast import *


class InterpolationMethod(Enum):
    Linear = 0
    Bilinear = 1


def plus(img1, img2):
    """
    Perform arithmetic operation plus, img1 + img2. Results that are greater than 1 get truncated to 1.
    :param img1: numpy array of greyscale image
    :param img2: numpy array of greyscale image
    :return: img1 + img2
    """
    if img1.shape != img2.shape:
        logger.error("plus: two operands have different shapes, returns img1")
        return img1

    img = img1 + img2
    img[img > 1] = 1
    return img


def minus(img1, img2):
    """
    Perform arithmetic operation minus, img1 - img2. Results that are less than 0 get truncated to 0.
    :param img1: numpy array of greyscale image
    :param img2: numpy array of greyscale image
    :return: img1 - img2
    """
    if img1.shape != img2.shape:
        logger.error("minus: two operands have different shapes, returns img1")
        return img1

    img = img1 - img2
    img[img < 0] = 0
    return img


def times(img1, img2):
    """
    Perform arithmetic operation multiplication, img1 * img2. Results that are greater than 1 get truncated to 1.
    :param img1: numpy array of greyscale image
    :param img2: numpy array of greyscale image
    :return: img1 - img2
    """
    if img1.shape != img2.shape:
        logger.error("times: two operands have different shapes, returns img1")
        return img1

    img = img1 * img2
    img[img > 1] = 1
    return img


def crop(img, point1, point2):
    """
    Crop image. Order of two points does not matter.
    Note: Cropped image shares data with the original one
    :param img: numpy array of greyscale/color image
    :param point1: one of the point of the selected region
    :param point2: another point at the opposite angle of the selected region
    :return: numpy array of cropped greyscale/color image
    """
    up, down = (point1[0], point2[0]) if point1[0] < point2[0] else (point2[0], point1[0])
    left, right = (point1[1], point2[1]) if point1[1] < point2[1] else (point2[1], point1[1])

    return img[up:down + 1, left:right + 1]


def resize(img, width, height, method=InterpolationMethod.Linear):
    """
    Resize image
    :param img: numpy array of greyscale/color image
    :param width: desired width
    :param height: desired height
    :param method: interpolation method, from enum InterpolationMethod
    :return: numpy array of resized greyscale/color image
    """

    if method not in InterpolationMethod:
        logger.warn("resize: unsupported interpolation method, linear interpolation is used")
        method = InterpolationMethod.Linear

    if is_greyscale(img):
        old_max_row_index, old_max_col_index = img.shape[0] - 1, img.shape[1] - 1
        new_max_row_index, new_max_col_index = height - 1, width - 1
        result = np.zeros((height, width))
        for row_index, row in enumerate(result):
            for col_index, pixel in enumerate(row):
                exact_mapped_row = row_index / new_max_row_index * old_max_row_index
                exact_mapped_col = col_index / new_max_col_index * old_max_col_index
                if method == InterpolationMethod.Linear:
                    result[row_index, col_index] = img[round(exact_mapped_row), round(exact_mapped_col)]
                elif method == InterpolationMethod.Bilinear:
                    row_0, col_0 = floor(exact_mapped_row), floor(exact_mapped_col)
                    if row_0 == old_max_row_index:  # in case of index out of range
                        row_0 -= 1
                    if col_0 == old_max_col_index:
                        col_0 -= 1
                    row_1, col_1 = row_0 + 1, col_0 + 1

                    y = np.array([img[row_0:row_1 + 1, col_0:col_1 + 1]]).reshape((4, 1))
                    w = np.array([[row_0, col_0, row_0 * col_0, 1],
                                  [row_0, col_1, row_0 * col_1, 1],
                                  [row_1, col_0, row_1 * col_0, 1],
                                  [row_1, col_1, row_1 * col_1, 1]])
                    x = np.linalg.inv(w).dot(y).reshape((1, 4))

                    result[row_index, col_index] = x.dot(
                        np.array([exact_mapped_row, exact_mapped_col, exact_mapped_row * exact_mapped_col, 1]))
    else:  # separately resize each channel
        result = np.zeros((height, width, 3))
        result[:, :, 0] = resize(img[:, :, 0], width, height, method)
        result[:, :, 1] = resize(img[:, :, 1], width, height, method)
        result[:, :, 2] = resize(img[:, :, 2], width, height, method)

    return result


def rotate(img, angle, method=InterpolationMethod.Linear, background="FFFFFF"):
    """
    Rotate image clockwise by given angle, and fill background with `background` color
    :param img: numpy array of greyscale/color image
    :param angle: rotation angle, within [0, 360)
    :param method: interpolation method, from enum InterpolationMethod
    :param background: background color
    :return: numpy array of rotated greyscale/color image
    """
    if method not in InterpolationMethod:
        logger.warn("rotate: unsupported interpolation method, linear interpolation is used")
        method = InterpolationMethod.Linear

    if is_greyscale(img):
        # rotate some 90 degrees and convert angle to radians
        angle = angle % 360
        img = np.rot90(img, -(angle // 90))
        angle = (angle % 90) / 180 * pi
        if angle == 0:
            return img

        old_height, old_width = img.shape
        old_max_row_index, old_max_col_index = img.shape[0] - 1, img.shape[1] - 1
        result = np.ones((round(old_height * cos(angle) + old_width * sin(angle)),
                          round(old_height * sin(angle) + old_width * cos(angle))))

        for row_index, row in enumerate(result):
            for col_index, pixel in enumerate(row):
                exact_mapped_row = row_index * cos(angle) - col_index * sin(angle) + old_height * sin(angle) ** 2
                exact_mapped_col = row_index * sin(angle) + col_index * cos(angle) - old_height * sin(angle) * cos(
                    angle)

                if method == InterpolationMethod.Linear:
                    old_row, old_col = round(exact_mapped_row), round(exact_mapped_col)
                    if not (old_row < 0 or old_row >= old_height or old_col < 0 or old_col >= old_width):
                        result[row_index, col_index] = img[old_row, old_col]
                elif method == InterpolationMethod.Bilinear:
                    row_0, col_0 = floor(exact_mapped_row), floor(exact_mapped_col)
                    if not (row_0 < 0 or row_0 >= old_height or col_0 < 0 or col_0 >= old_width):
                        if row_0 == old_max_row_index:  # in case of index out of range
                            row_0 -= 1
                        if col_0 == old_max_col_index:
                            col_0 -= 1

                        row_1, col_1 = row_0 + 1, col_0 + 1

                        y = np.array([img[row_0:row_1 + 1, col_0:col_1 + 1]]).reshape((4, 1))
                        w = np.array([[row_0, col_0, row_0 * col_0, 1],
                                      [row_0, col_1, row_0 * col_1, 1],
                                      [row_1, col_0, row_1 * col_0, 1],
                                      [row_1, col_1, row_1 * col_1, 1]])
                        x = np.linalg.inv(w).dot(y).reshape((1, 4))

                        result[row_index, col_index] = x.dot(
                            np.array([exact_mapped_row, exact_mapped_col, exact_mapped_row * exact_mapped_col, 1]))
    else:
        b = np.expand_dims(rotate(img[:, :, 0], angle, method), 2)
        g = np.expand_dims(rotate(img[:, :, 1], angle, method), 2)
        r = np.expand_dims(rotate(img[:, :, 2], angle, method), 2)
        result = np.concatenate((b, g, r), axis=2)

    return result
