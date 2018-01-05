import numpy as np

from math import sqrt, ceil, exp
from .util import *

from .basic import *


class KernelShape(Enum):
    Circle = 0
    Square = 1


def kernel_circle_mask(height, width):
    """
    Provide mask for generating circle kernel.
    :param height: kernel height
    :param width: kernel width
    :return: 2d numpy array with 1's at where is inside the circle
    """
    kernel = np.zeros((height, width))
    for col_index in range(width):
        w = col_index + 0.5
        h1 = height * (1 - sqrt(1 - (2 * w / width - 1) ** 2)) / 2
        h2 = height * (1 + sqrt(1 - (2 * w / width - 1) ** 2)) / 2
        for row_index in range(int(ceil(h1 - 0.5)), int(ceil(h2 - 0.49))):  # -0.49 to include points that on the circle
            kernel[row_index, col_index] = 1

    return kernel


def gaussian_func(x, y, variance):
    """
    Return value of Gaussian function at (x, y)
    :param x: x coordinate
    :param y: y coordinate
    :param variance: variance of Gaussian distribution
    :return: value of gaussian function at (x, y)
    """
    return exp(-(x ** 2 + y ** 2) / (2 * variance))


def kernel_mean(height, width, shape=KernelShape.Square):
    """
    Generate kernel for mean filter
    :param height: kernel height
    :param width: kernel width
    :param shape: kernel shape
    :return: kernel for mean filter
    """
    if height % 2 != 1 or width % 2 != 1:
        logger.error("kernel_mean: kernel height and width should be odd")
        return np.array([[1]])

    if shape not in KernelShape:
        logger.warn("kernel_mean: unsupported kernel shape, square is used")
        shape = KernelShape.Square

    if shape == KernelShape.Square:
        kernel = np.ones((height, width))
    elif shape == KernelShape.Circle:
        kernel = kernel_circle_mask(height, width)
    else:
        kernel = None  # just make IDE happy

    return kernel / kernel.sum()


def kernel_gaussian(height, width, variance=None, shape=KernelShape.Square):
    """
    Generate kernel for gaussian filter
    :param height: kernel height
    :param width: kernel width
    :param variance: variance of Gaussian distribution
    :param shape: kernel shape
    :return: kernel for gaussian filter
    """
    if height % 2 != 1 or width % 2 != 1:
        logger.error("kernel_gaussian: kernel height and width should be odd")
        return np.array([[1]])

    if shape not in KernelShape:
        logger.warn("kernel_gaussian: unsupported kernel shape, square is used")
        shape = KernelShape.Square

    if variance is None:
        sigma = 0.3 * (((height + width) / 2 - 1) * 0.5 - 1) + 0.8
        variance = sigma ** 2

    if shape == KernelShape.Square:
        kernel = np.ones((height, width))
    elif shape == KernelShape.Circle:
        kernel = kernel_circle_mask(height, width)
    else:
        kernel = None  # just make IDE happy

    for row_index in range(height):
        for col_index in range(width):
            if kernel[row_index, col_index] != 0:
                x = col_index - (width - 1) / 2
                y = row_index - (height - 1) / 2
                kernel[row_index, col_index] = gaussian_func(x, y, variance)

    return kernel / kernel.sum()


def mean_filter(img, kernel_height, kernel_width, kernel_shape=KernelShape.Square):
    """
    Wrap for mean filter, perform mean filtering.
    :param img: numpy array of greyscale image
    :param kernel_width: kernel width
    :param kernel_height: kernel height
    :param kernel_shape: kernel shape
    :return: numpy array of filtered greyscale image
    """
    if kernel_shape not in KernelShape:
        logger.warn("mean_filter: unsupported kernel shape, square is used")
        kernel_shape = KernelShape.Square

    return convolve(img, kernel_mean(kernel_height, kernel_width, kernel_shape))


def median_filter(img, kernel_height, kernel_width, kernel_shape=KernelShape.Square):
    """
    Perform median filtering.
    :param img: numpy array of greyscale image
    :param kernel_width: kernel width
    :param kernel_height: kernel height
    :param kernel_shape: kernel shape
    :return: numpy array of filtered greyscale image
    """
    if kernel_shape not in KernelShape:
        logger.warn("median_filter: unsupported kernel shape, square is used")
        kernel_shape = KernelShape.Square

    if kernel_shape == KernelShape.Square:
        kernel = np.ones((kernel_height, kernel_width))

        def get_median(roi):
            return np.median(roi)
    elif kernel_shape == KernelShape.Circle:
        kernel = kernel_circle_mask(kernel_height, kernel_width)
        flat_bool = kernel.astype(np.bool).flatten()

        def get_median(roi):
            return np.median(roi[flat_bool])
    else:
        kernel, get_median = None, None  # just to make IDE happy

    result = convolve(img, kernel, kernel_func=get_median)
    return result


def gaussian_filter(img, kernel_height, kernel_width, variance=None, kernel_shape=KernelShape.Square):
    """
    Wrap for gaussian filter, perform mean filtering.
    :param img: numpy array of greyscale image
    :param kernel_width: kernel width
    :param kernel_height: kernel height
    :param kernel_shape: kernel shape
    :param variance: variance of gaussian distribution
    :return: numpy array of filtered greyscale image
    """
    if kernel_shape not in KernelShape:
        logger.warn("mean_filter: unsupported kernel shape, square is used")
        kernel_shape = KernelShape.Square

    return convolve(img, kernel_gaussian(kernel_height, kernel_width, variance, kernel_shape))


# TODO: allow different size
def laplacian_filter(img):
    """
    Perform Laplacian filtering on img
    :param img: greyscale image
    :return: laplacian of image, un-normalized
    """

    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return convolve(img, kernel)
