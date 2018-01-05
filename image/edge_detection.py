import numpy as np

from math import pi

from .util import *
from .filtering import *


def LoG_func(x, y, variance):
    """
    Return value of Laplacian-of-Gaussian at (x, y)
    :param x: x coordinate
    :param y: y coordinate
    :param variance: variance of Gaussian
    :return: LoG(x, y)
    """
    return (x ** 2 + y ** 2 - 2 * variance) / (variance ** 2) * gaussian_func(x, y, variance)


def kernel_LoG(size, variance=None):
    """
    Generate kernel for LoG filter
    :param size: kernel size
    :param variance: variance of Gaussian distribution
    :return: kernel for LoG filter
    """
    if size % 2 != 1:
        logger.error("kernel_LoG: kernel size must odd")
        # return a defult one
        return np.array([[0, 0, -1, 0, 0],
                         [0, -1, -2, -1, 0],
                         [-1, -2, 16, -2, -1],
                         [0, -1, -2, -1, 0],
                         [0, 0, -1, 0, 0]])

    if variance is None:
        variance = (size / 6) ** 2

    kernel = np.ones((size, size))
    for row_index in range(size):
        for col_index in range(size):
            if kernel[row_index, col_index] != 0:
                x = col_index - (size - 1) / 2
                y = row_index - (size - 1) / 2
                kernel[row_index, col_index] = LoG_func(x, y, variance)

    return kernel


def kernel_grad_x(size=3):
    """
    Return kernel for x gradient
    :return: kernel for x gradient
    """
    return np.array([[3, 0, -3],
                     [10, 0, -10],
                     [3, 0, -3]])


def kernel_grad_y(size=3):
    return np.rot90(kernel_grad_x(size), -1)


def zero_crossing(img, threshold=0):
    """
    Mark the zero crossing of (gradient) img
    :param img: greyscale img
    :param threshold: for filtering cneter pixel
    :return: binary img where zero crossings are marked 1, otherwise 0
    """

    size = 3

    def pick(roi):
        roi = roi.reshape((size, size))
        roi_rot180 = np.rot90(roi, 2)
        mul = (roi * roi_rot180).reshape(-1)  # element-wise mul
        if (mul[0] < 0 or mul[1] < 0 or mul[2] < 0 or mul[3] < 0) and abs(roi[1, 1]) > threshold:
            return roi[1, 1]
        else:
            return 0.0

    return convolve(img, np.ones((size, size)), kernel_func=pick)


def sobel(img, size=3, true_grad=False):
    """
    Return gradient image produced by Sobel edge detection.
    :param img: numpy array of greyscale image
    :param true_grad: whether to use (g_x ** 2 + g_y ** 2) ** 0.5 for final result
    :return: gradient value of img
    """
    if not is_greyscale(img):
        logger.warn("sobel: input img should be greyscale")
        return img

    grad_x = convolve(img, kernel_grad_x(size))
    grad_y = convolve(img, kernel_grad_y(size))

    if true_grad:
        result = np.sqrt(grad_x ** 2 + grad_y ** 2)
    else:
        result = abs(grad_x) + abs(grad_y)

    return result


def marr_hildreth(img, size=5, threshold=0):
    """
    Perform Marr-Hildreth edge detection
    :param img: greyscale img
    :param size: kelnel size, must be odd
    :param threshold: threshold for zero-crossing finding
    :return: edge detection result
    """
    kernel = kernel_LoG(size)
    lap_o_gaus = convolve(img, kernel)
    result = zero_crossing(lap_o_gaus, threshold)
    return result


def canny(img, threshold_lo, threshold_hi, size=3):
    """
    Perform Canny edge detection
    :param img: greyscale image
    :param threshold_lo: low threshold in canny algorithm
    :param threshold_hi: high threshold in canny algorithm
    :param size: kernel size
    :return: edge detection result
    """
    # Gaussian blur
    img = gaussian_filter(img, size, size)

    # gradients
    grad_x = convolve(img, kernel_grad_x(size))
    grad_y = convolve(img, kernel_grad_y(size))
    mag_g = normalize(np.sqrt(grad_x ** 2 + grad_y ** 2))  # make sure mag is within [0, 1]
    angle_g = np.arctan(grad_y / (grad_x + 0.0001))  # constant added to avoid zero division error

    # non-maximum suppression
    img = np.zeros(img.shape)
    row_num, col_num = img.shape
    for row_i in range(row_num):
        for col_i in range(col_num):
            angle = angle_g[row_i, col_i]
            # noinspection PyTypeChecker
            if angle < - 3 * pi / 8 or angle >= 3 * pi / 8:  # vertical
                center = mag_g[row_i, col_i]
                neighbour = []
                if row_i != 0:
                    neighbour.append(mag_g[row_i - 1, col_i])
                if row_i != row_num - 1:
                    neighbour.append(mag_g[row_i + 1, col_i])
                if center >= max(neighbour):
                    img[row_i, col_i] = center
            elif - 3 * pi / 8 <= angle < - pi / 8:  # leftdown-rightup
                center = mag_g[row_i, col_i]
                neighbour = [0]  # to avoid empty list in two corners
                if row_i != 0 and col_i != col_num - 1:
                    neighbour.append(mag_g[row_i - 1, col_i + 1])
                if row_i != row_num - 1 and col_i != 0:
                    neighbour.append(mag_g[row_i + 1, col_i - 1])
                if center >= max(neighbour):
                    img[row_i, col_i] = center
            elif - pi / 8 <= angle < pi / 8:  # horizontal
                center = mag_g[row_i, col_i]
                neighbour = []
                if col_i != 0:
                    neighbour.append(mag_g[row_i, col_i - 1])
                if col_i != col_num - 1:
                    neighbour.append(mag_g[row_i, col_i + 1])
                if center >= max(neighbour):
                    img[row_i, col_i] = center
            else:  # leftup-rightdown
                center = mag_g[row_i, col_i]
                neighbour = [0]
                if row_i != 0 and col_i != 0:
                    neighbour.append(mag_g[row_i - 1, col_i - 1])
                if row_i != row_num - 1 and col_i != col_num - 1:
                    neighbour.append(mag_g[row_i + 1, col_i + 1])
                if center >= max(neighbour):
                    img[row_i, col_i] = center

    # double threshold and edge tracking
    lo_g = np.ones(img.shape, np.bool)
    lo_g[img < threshold_lo] = False
    lo_g[img >= threshold_hi] = False
    hi_g = np.ones(img.shape, np.bool)
    hi_g[img < threshold_hi] = False

    lo_visited = np.zeros(img.shape, np.bool)
    lo_to_select = np.zeros(img.shape, np.bool)
    lo_selected = np.zeros(img.shape, np.bool)

    def visit_lo(row_i, col_i):
        lo_visited[row_i, col_i] = True

        # visit forward or add a lo pixel to to_select list
        if row_i != 0:
            if lo_g[row_i - 1, col_i] and not lo_visited[row_i - 1, col_i]:
                visit_lo(row_i - 1, col_i)
            elif hi_g[row_i - 1, col_i]:
                lo_to_select[row_i, col_i] = True

        if row_i != 0 and col_i != col_num - 1:
            if lo_g[row_i - 1, col_i + 1] and not lo_visited[row_i - 1, col_i + 1]:
                visit_lo(row_i - 1, col_i + 1)
            elif hi_g[row_i - 1, col_i + 1]:
                lo_to_select[row_i, col_i] = True

        if row_i != row_num - 1:
            if lo_g[row_i + 1, col_i] and not lo_visited[row_i + 1, col_i]:
                visit_lo(row_i + 1, col_i)
            elif hi_g[row_i + 1, col_i]:
                lo_to_select[row_i, col_i] = True

        if row_i != row_num - 1 and col_i != 0:
            if lo_g[row_i + 1, col_i - 1] and not lo_visited[row_i + 1, col_i - 1]:
                visit_lo(row_i + 1, col_i - 1)
            elif hi_g[row_i + 1, col_i - 1]:
                lo_to_select[row_i, col_i] = True

        if col_i != 0:
            if lo_g[row_i, col_i - 1] and not lo_visited[row_i, col_i - 1]:
                visit_lo(row_i, col_i - 1)
            elif hi_g[row_i, col_i - 1]:
                lo_to_select[row_i, col_i] = True

        if col_i != 0 and row_i != 0:
            if lo_g[row_i - 1, col_i - 1] and not lo_visited[row_i - 1, col_i - 1]:
                visit_lo(row_i - 1, col_i - 1)
            elif hi_g[row_i - 1, col_i - 1]:
                lo_to_select[row_i, col_i] = True

        if col_i != col_num - 1:
            if lo_g[row_i, col_i + 1] and not lo_visited[row_i, col_i - 1]:
                visit_lo(row_i, col_i + 1)
            elif hi_g[row_i, col_i + 1]:
                lo_to_select[row_i, col_i] = True

        if col_i != col_num - 1 and row_i != row_num - 1:
            if lo_g[row_i + 1, col_i + 1] and not lo_visited[row_i + 1, col_i - 1]:
                visit_lo(row_i + 1, col_i + 1)
            elif hi_g[row_i + 1, col_i + 1]:
                lo_to_select[row_i, col_i] = True

    def select_lo(row_i, col_i):
        hi_g[row_i, col_i] = True
        lo_selected[row_i, col_i] = True

        # visit forward or add a lo pixel to to_select list
        if row_i != 0:
            if lo_g[row_i - 1, col_i] and not lo_selected[row_i - 1, col_i]:
                select_lo(row_i - 1, col_i)

        if row_i != 0 and col_i != col_num - 1:
            if lo_g[row_i - 1, col_i + 1] and not lo_selected[row_i - 1, col_i + 1]:
                select_lo(row_i - 1, col_i + 1)

        if row_i != row_num - 1:
            if lo_g[row_i + 1, col_i] and not lo_selected[row_i + 1, col_i]:
                select_lo(row_i + 1, col_i)

        if row_i != row_num - 1 and col_i != 0:
            if lo_g[row_i + 1, col_i - 1] and not lo_selected[row_i + 1, col_i - 1]:
                select_lo(row_i + 1, col_i - 1)

        if col_i != 0:
            if lo_g[row_i, col_i - 1] and not lo_selected[row_i, col_i - 1]:
                select_lo(row_i, col_i - 1)

        if col_i != 0 and row_i != 0:
            if lo_g[row_i - 1, col_i - 1] and not lo_selected[row_i - 1, col_i - 1]:
                select_lo(row_i - 1, col_i - 1)

        if col_i != col_num - 1:
            if lo_g[row_i, col_i + 1] and not lo_selected[row_i, col_i + 1]:
                select_lo(row_i, col_i + 1)

        if col_i != col_num - 1 and row_i != row_num - 1:
            if lo_g[row_i + 1, col_i + 1] and not lo_selected[row_i + 1, col_i - 1]:
                select_lo(row_i + 1, col_i + 1)

    # first visit all lo segs and mark those are connected to hi segs
    for row_i in range(row_num):
        for col_i in range(col_num):
            if lo_g[row_i, col_i] and not lo_visited[row_i, col_i]:
                visit_lo(row_i, col_i)

    # the select those mark lo seg to the result
    # NOTE: hi_g is now treated as the final select result
    for row_i in range(row_num):
        for col_i in range(col_num):
            if lo_to_select[row_i, col_i] and not lo_selected[row_i, col_i]:
                select_lo(row_i, col_i)

    # Done!
    return normalize(hi_g.astype(np.float))
