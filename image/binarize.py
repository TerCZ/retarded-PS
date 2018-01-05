import numpy as np

from math import pi

from .util import *
from .contrast import get_histogram


def threshold(img, thresh):
    """
    Perform basic binarization with fixed threshold
    :param img: greyscale image
    :param thresh: threshold within [0, 1]
    :return: binary image
    """
    if not is_greyscale(img):
        logger.error("threshold: img must be greyscale")
        return img

    if not 0 <= thresh <= 1:
        logger.warn("threshold: thresh should be within [0, 1]")

    result = np.zeros(img.shape)
    result[img >= thresh] = 1
    return result


def threshold2(img, thresh_lo, thresh_hi):
    """
    Perform basic binarization with fixed threshold, pixel in [lo, hi) are kept
    :param img: greyscale image
    :param thresh_lo: threshold within [0, 1]
    :param thresh_hi: threshold within [0, 1]
    :return: binary image
    """
    if not is_greyscale(img):
        logger.error("threshold: img must be greyscale")
        return img

    if not 0 <= thresh_lo <= thresh_hi <= 1:
        logger.warn("threshold: thresh should be within [0, 1], and thresh_hi >= shresh_lo")

    result = np.ones(img.shape)
    result[img < thresh_lo] = 0
    result[img >= thresh_hi] = 0
    return result


def otus(img, bins=256):
    """
    Perform Otus threshold on greysclae image
    :param img: greyscale image
    :param bins: number of thresholds to be tested
    :return: binary result
    """
    if bins <= 0:
        logger.error("otus: bins must be positive")
        return img

    if not is_greyscale(img):
        logger.error("otus: img must be greyscale")
        return img

    hist = get_histogram(img, bins)
    p_1 = np.array([hist[:k + 1].sum() for k in range(bins)])
    mean_1 = np.array([hist[:k + 1].dot(np.arange(0, k + 1)) for k in range(bins)])
    for k in range(bins):
        if p_1[k] != 0:
            mean_1[k] /= p_1[k]
    mean_g = hist.dot(np.arange(0, bins))
    variance = np.zeros((bins,))
    for k in range(bins):
        if p_1[k] != 1:
            variance[k] = p_1[k] / (1 - p_1[k]) * (mean_1[k] - mean_g) ** 2
            if variance[k] == np.nan:
                print(p_1[k], mean_1[k], mean_g)

    max_var = variance.max()
    matches = [k for k, var in enumerate(variance) if var == max_var]
    k = sum(matches) / len(matches)
    thresh = (k + 1) / (bins - 1)  # kept [k+1, L-1] part

    return threshold(img, thresh)
