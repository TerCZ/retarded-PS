import cv2
import numpy as np
import logging

from time import time

logger = logging.getLogger("Retarted-PS")
logger.setLevel(logging.INFO)


def timeit(func):
    def wrapped(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print("{} used {:.5} sec".format(func.__name__, time() - start))
        return result

    return wrapped


@timeit
def convolve(img, kernel, kernel_func=None, normalize_kernel=False, normalize_result=False):
    """
    Perform 2d convolution
    :param img: first input 2d numpy array
    :param kernel: second input 2d numpy array
    :return: convolution result, with the same size as img
    """
    if not is_greyscale(kernel):
        logger.error("convolve: kernel should be 2d")
        return img

    if not is_greyscale(img):
        b = np.expand_dims(convolve(img[:, :, 0], kernel, kernel_func, normalize_kernel, normalize_result), 2)
        g = np.expand_dims(convolve(img[:, :, 1], kernel, kernel_func, normalize_kernel, normalize_result), 2)
        r = np.expand_dims(convolve(img[:, :, 2], kernel, kernel_func, normalize_kernel, normalize_result), 2)
        return np.concatenate((b, g, r), axis=2)
    else:
        if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
            logger.error("convolve: kernel length should be odd")
            return img

        if normalize_kernel:
            kernel = kernel / kernel.sum()

        # rotate kernel
        kernel = np.rot90(kernel, 2)
        kernel_height, kernel_width = kernel.shape

        # pad the edges with 0, I guess
        padded = np.pad(img, (((kernel_height - 1) // 2, (kernel_height - 1) // 2),  # TODO: -1 necessary?
                              ((kernel_width - 1) // 2, (kernel_width - 1) // 2)), "constant", constant_values=1)

        flat_kernel = kernel.flatten()

        result = np.zeros(img.shape, dtype=img.dtype)
        if kernel_func is None:
            for row_index in range(result.shape[0]):
                for col_index in range(result.shape[1]):
                    roi = padded[row_index:row_index + kernel_height, col_index:col_index + kernel_width].reshape(-1)
                    result[row_index, col_index] = flat_kernel.dot(roi)
        else:
            for row_index in range(result.shape[0]):
                for col_index in range(result.shape[1]):
                    roi = padded[row_index:row_index + kernel_height, col_index:col_index + kernel_width].reshape(-1)
                    result[row_index, col_index] = kernel_func(roi)

        if normalize_result:
            result = normalize(result)

        return result


def iter_pixel(img, func):
    result = np.zeros(img.shape)
    for row_index, row in enumerate(img):
        for col_index, pixel in enumerate(row):
            result[row_index, col_index] = func(pixel)

    return result


def normalize(img):
    """
    Normalize pixel density by linear mapping them to [0, 1]
    :param img: numpy array of un-regularized greyscale image, in greyscale
    :return: numpy array of regularized greyscale image, in greyscale
    """
    max_val = img.max()
    min_val = img.min()

    if min_val == max_val:
        return img

    def transform(pixel):
        x1, y1 = (min_val, 0)
        x2, y2 = (max_val, 1)
        return (y2 - y1) / (x2 - x1) * (pixel - x1) + y1

    return iter_pixel(img, transform)


def to8bit(img):
    return (img*255).astype(np.int8)


def is_greyscale(img):
    """
    Check if image is in greyscale format
    :param img: numpy array of image
    :return: True for greyscale, else False
    """
    return img.ndim == 2


def is_binary(img):
    """
    Check if image is binary
    :param img: a greyscale img
    :return: True for binary, else False
    """
    return is_greyscale(img) and not np.any(img[img != 0] != 1)


def show_grey(img):
    """
    Shortcut to preview greyscale image
    :param img: numpy array of greyscale image
    :return: nothing
    """

    # convert for OpenCV
    img = img.copy()
    img[:, :] *= 255
    img = img.astype(np.uint8)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_color(img):
    """
    Shortcut to preview color image
    :param img: numpy array of color image, in BGR mode
    :return: nothing
    """

    # convert for OpenCV
    img = img.copy()
    img[:, :, :] *= 255
    img = img.astype(np.uint8)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
