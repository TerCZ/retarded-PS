import numpy as np

from math import log1p
from .util import *

from .basic import *


def linear_mapping(img, anchors):
    """
    Perform linear density mapping on greyscale image.
    Example: with anchors = [(0,0), (0.5, 0.25), (1,1)], density of pixel between [0, 0.5] will be mapped to [0, 0.25],
    and [0.5, 1] mapped to [0.25, 1]
    Note: the first/last element of anchors should have 0/1 as x coordinate
    :param img: numpy array of greyscale image
    :param anchors: list of end points of the lines in density transformation graph, in [0, 1]^2 space
    :return: numpy array of transformed greyscale image
    """
    if not is_greyscale(img):
        logger.warning("linear_mapping: try to do mapping on color image")
        return img

    if len(anchors) < 2:
        logger.warning("linear_mapping: at least 2 anchors are needed")
        return img

    for point in anchors:
        if point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1:
            logger.warning("linear_mapping: element of src/dst should be within [0, 1]")
            return img

    # sort anchors by x coordinates
    anchors.sort(key=lambda p: p[0])

    if anchors[0][0] != 0 or anchors[-1][0] != 1:
        logger.warning("linear_mapping: 0 and 1 should be the first and the last element of src list")
        return img

    def transform(pixel):
        anchor_index = 0  # just make IDE happy
        for anchor_index, point in enumerate(anchors):
            if point[0] > pixel:
                break

        x1, y1 = anchors[anchor_index - 1]
        x2, y2 = anchors[anchor_index]
        return (y2 - y1) / (x2 - x1) * (pixel - x1) + y1

    return iter_pixel(img, transform)


def log_mapping(img, c):
    """
    Perform logarithm transformation on greyscale image, using s = c * log(1 + r).
    Results are regularized (linearly mapping to [0, 1]).
    :param img: numpy array of greyscale image
    :param c: parameter in s = c * log(1 + r)
    :return: numpy array of transformed greyscale image
    """

    if not is_greyscale(img):
        logger.warning("log_mapping: try to do mapping on color image")
        return img

    unregularized = np.log1p(img) * c
    return normalize(unregularized)


def power_mapping(img, c, g):
    """
    Perform power transformation on greyscale image, using s = c * r ^ g.
    Results are regularized (linearly mapping to [0, 1]).
    :param img: numpy array of greyscale image
    :param c: parameter in s = c * r ^ g
    :param g: parameter in s = c * r ^ g
    :return: numpy array of transformed greyscale image
    """
    if not is_greyscale(img):
        logger.warning("power_mapping: try to do mapping on color image")
        return img

    unregularized = np.power(img, g) * c
    return normalize(unregularized)


def get_histogram_int(img, bins=256):
    """
    Calculate histogram of given image.
    :param img: numpy array of greyscale image
    :param bins: number of equal-width bins in histogram, default to 256
    :return: the values of the histogram
    """
    if not is_greyscale(img):
        logger.warning("get_histogram: try to do mapping on color image")
        return img

    result = np.zeros((bins,), np.int64)

    def statistic(pixel):
        result[pixel] += 1

    iter_pixel((img * (bins - 1)).astype(np.uint), statistic)
    return result


def get_histogram(img, bins=256):
    """
    Calculate histogram of given image.
    :param img: numpy array of greyscale image
    :param bins: number of equal-width bins in histogram, default to 256
    :return: the values of the histogram
    """
    if not is_greyscale(img):
        logger.warning("get_histogram: try to do mapping on color image")
        return img

    result = get_histogram_int(img)
    result = result / result.sum()  # REVIEW: don't know if this will break hist_equalize
    return result


def histogram_qualization(img):
    data = img.flatten()
    data.sort()
    rank = {}
    for index, density in enumerate(data):
        rank[density] = index

    max_val = img.max()
    size = img.size

    def transform(pixel):
        return max_val * rank[pixel] / size

    return iter_pixel(img, transform)
