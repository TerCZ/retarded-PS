import numpy as np

from collections import deque
from enum import Enum
from math import pi

from .util import *
from .basic import *


class MorphMethod(Enum):
    Erode = 0
    Dilate = 1
    Open = 2
    Close = 3


def morph_binary(img, kernel, method):
    """
    Wrapper for binary morph operations
    :param img: binary image
    :param kernel: morph kernel
    :param method: morph method
    :return: binary image
    """
    if not is_binary(img) or not is_binary(kernel):
        logger.error("morph_binary: img/kernel should be binary")
        return img

    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        logger.error("morph_binary: kernel length/width should be odd")
        return img

    return morph(img, kernel, method, True)


def morph_greyscale(img, kernel, method, flat_kernel=True):
    """
    Wrapper for greyscale morph operation
    :param img: greyscale image
    :param kernel: morph kernel
    :param method: morph method
    :param flat_se: whether kernel is flat
    :return: morphed image
    """
    if not is_greyscale(img) or not is_greyscale(kernel):
        logger.error("morph_greyscale: img/kernel should be greyscale")
        return img

    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        logger.error("morph_greyscale: kernel length/width should be odd")
        return img

    return morph(img, kernel, method, flat_kernel)


def morph(img, kernel, method, flat_kernel=True):
    if method == MorphMethod.Erode:
        return erode(img, kernel, flat_kernel)
    elif method == MorphMethod.Dilate:
        return dilate(img, kernel, flat_kernel)
    elif method == MorphMethod.Open:
        return dilate(erode(img, kernel, flat_kernel), kernel, flat_kernel)
    elif method == MorphMethod.Close:
        return erode(dilate(img, kernel, flat_kernel), kernel, flat_kernel)
    else:
        logger.error("morph: unknown method:", method)
        return img


def erode(img, kernel, flat_kernel=True):
    """
    Perform erosion on img
    :param img: binary image
    :param kernel: kernel
    :param flat_kernel: whether kernel is flat
    :return: erosion result
    """

    # NOTE: roi received is not rotated
    if flat_kernel:
        def erode_pixel(roi):
            roi = roi.reshape(kernel.shape)
            return roi[kernel != 0].min()
    else:
        def erode_pixel(roi):
            roi = roi.reshape(kernel.shape)
            return (roi - kernel).min()  # a rectangle kernel is assumed

    return convolve(img, kernel, erode_pixel)


def dilate(img, kernel, flat_kernel=True):
    """
    Perform dilation on img
    :param img: binary image
    :param kernel: kernel
    :param flat_kernel: whether kernel is flat
    :return: dilation result
    """

    kernel = np.rot90(kernel, 2)  # NOTE: roi received is not rotated
    if flat_kernel:
        def erode_pixel(roi):
            roi = roi.reshape(kernel.shape)
            return roi[kernel != 0].max()
    else:
        def erode_pixel(roi):
            roi = roi.reshape(kernel.shape)
            return (roi + kernel).max()

    return convolve(img, kernel, erode_pixel)


def thickening(img, kernel, iter_num=None):
    """
    Perform thickening algorithm on binary image
    :param img: binary image
    :param kernel: kernel for thickening, 0.5 for don't care
    :param iter_num: number of iteration, None for repeating until converge
    :return: binary skeleton
    """
    if not is_binary(img):
        logger.error("thickening: img should be binary")
        return img

    if np.any(kernel[np.bitwise_and(kernel != 0, kernel != 1)] != 0.5):
        logger.error("thickening: kernel value for thinning should be 0, 0.5 or 1")
        return img

    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        logger.error("thickening: kernel should have odd height/width")
        return img

    if iter_num is None:
        iter_num = -1

    def thin_pixel(roi):
        if not np.any(roi.reshape(kernel.shape)[kernel != 0.5] != kernel[kernel != 0.5]):  # a hit
            return 1
        else:
            return roi[kernel.size // 2]

    iter = 0
    while iter != iter_num:
        logger.info("thickening: iter", iter)
        print("thickening: iter", iter)
        new_img = convolve(img, kernel, kernel_func=thin_pixel)

        if not np.any(img != new_img):
            return new_img

        iter += 1
        img = new_img

    return img


def thinning(img, kernel, iter_num=None):
    """
    Perform thinning algorithm on binary image
    :param img: binary image
    :param kernel: kernel for thinning, 0.5 for don't care
    :param iter_num: number of iteration, None for repeating until converge
    :return: binary skeleton
    """
    if not is_binary(img):
        logger.error("thinning: img should be binary")
        return img

    if np.any(kernel[np.bitwise_and(kernel != 0, kernel != 1)] != 0.5):
        logger.error("thinning: kernel value for thinning should be 0, 0.5 or 1")
        return img

    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        logger.error("thinning: kernel should have odd height/width")
        return img

    if iter_num is None:
        iter_num = -1

    def thin_pixel(roi):
        if not np.any(roi.reshape(kernel.shape)[kernel != 0.5] != kernel[kernel != 0.5]):  # a hit
            return 0
        else:
            return roi[kernel.size // 2]

    iter = 0
    while iter != iter_num:
        logger.info("thinning: iter", iter)
        print("thinning: iter", iter)
        new_img = convolve(img, kernel, kernel_func=thin_pixel)

        if not np.any(img != new_img):
            return new_img

        iter += 1
        img = new_img

    return img


def thinning_skeleton(img, iter_num=None):
    """
    Perform thinning algorithm on binary image
    :param img: binary image
    :param kernel: kernel for thinning, 0.5 for don't care
    :param iter_num: number of iteration, None for repeating until converge
    :return: binary skeleton
    """
    if not is_binary(img):
        logger.error("thinning: img should be binary")
        return img

    if iter_num is None:
        iter_num = -1

    height_src, width_src = img.shape
    img = np.pad(img, ((1, 1), (1, 1)), "constant")

    kernel_b1 = np.array([[0, 0, 0], [0.5, 1, 0.5], [1, 1, 1]])
    kernel_b2 = np.array([[0, 0, 0.5], [0, 1, 1], [0.5, 1, 0.5]])
    kernels = [kernel_b1, kernel_b2, np.rot90(kernel_b1), np.rot90(kernel_b2),
               np.rot90(kernel_b1, 2), np.rot90(kernel_b2, 2), np.rot90(kernel_b1, 3), np.rot90(kernel_b2, 33)]

    iter = 0
    while iter != iter_num:
        start = time()
        logger.info("thinning: iter", iter)
        print("thinning: iter", iter)

        changed = False
        for kernel in kernels:
            for row_i in range(height_src):
                for col_i in range(width_src):
                    roi = img[row_i:row_i + 3, col_i:col_i + 3]
                    if not np.any(roi[kernel != 0.5] != kernel[kernel != 0.5]):  # a hit
                        img[row_i + 1, col_i + 1] = 0
                        changed = True

        iter += 1
        print("one round of thinning used {:.5} sec".format(time() - start))
        if not changed:
            break

    return img[1:-1, 1:-1]


def skeleton(img, kernel=None):
    """
    Perform skeleton algorithm
    :param img: binary image
    :param kernel: kernel
    :return: skeleton of image, and list of (s_k, k) pair for reconstructing
    """
    if kernel is None:
        kernel = np.ones((3, 3))

    img = img.astype(np.int)

    erode_ks = []
    while np.any(img):
        erode_ks.append(img)
        img = morph_binary(img, kernel, MorphMethod.Erode)

    # now img is all-zero
    s_ks = []
    for k, eroded in enumerate(erode_ks):
        s_k = eroded - morph_binary(eroded, kernel, MorphMethod.Open)
        s_ks.append((s_k, k))
        img = np.bitwise_or(img, s_k)

    return img, s_ks


def skeleton_reconstruct(sks, kernel=None):
    """
    Reconstruct skeleton from list of (s_k, k) pair for reconstructing
    :param sks: list of (s_k, k) pair for reconstructing, return from function skeleton
    :return: original image
    """
    if kernel is None:
        kernel = np.ones((3, 3))

    result = np.zeros(sks[0][0].shape, dtype=np.int)

    for s_k, k in sks:
        dilate_k = s_k
        for _ in range(k):
            dilate_k = morph_binary(dilate_k, kernel, MorphMethod.Dilate)
        result = np.bitwise_or(result, dilate_k)

    return result


def morphological_reconstruct_greyscale(marker, reference, kernel=None, method=None):
    """
    Perform morphological reconstruction
    :param marker: marker image
    :param reference: reference image
    :param reference: kernel for reconstruction
    :param method: MorphMethod.Erose or MorphMethod.Dilate
    :return: reconstruction result
    """
    if not is_greyscale(marker) or not is_greyscale(reference):
        logger.error("morphological_reconstruct: input image should be greyscale")
        return marker

    if kernel is None:
        kernel = np.ones((3, 3))

    if method == MorphMethod.Erode:
        restrict_func = np.maximum
    elif method == MorphMethod.Dilate:
        restrict_func = np.minimum
    else:
        logger.error("morphological_reconstruct: unsupported method:", method)
        return marker

    img = marker
    i = 0
    while True:
        new_img = morph_greyscale(img, kernel, method, True)
        new_img = restrict_func(new_img, reference)

        write("mid{}.png".format(i), new_img)

        if not np.any(new_img != img):
            return new_img

        img = new_img
        i += 1


def morphological_reconstruct_binary(marker, reference, kernel=None, method=None):
    """
    Perform morphological reconstruction
    :param marker: marker image
    :param reference: reference image
    :param reference: kernel for reconstruction
    :param method: MorphMethod.Erose or MorphMethod.Dilate
    :return: reconstruction result
    """
    if not is_binary(marker) or not is_binary(reference):
        logger.error("morphological_reconstruct_binary: input image should be binary")
        return marker

    if kernel is None:
        kernel = np.ones((3, 3))

    if method == MorphMethod.Erode:
        restrict_func = np.maximum
    elif method == MorphMethod.Dilate:
        restrict_func = np.minimum
    else:
        logger.error("morphological_reconstruct: unsupported method:", method)
        return marker

    img = marker
    while True:
        new_img = morph_binary(img, kernel, method)
        new_img = restrict_func(new_img, reference)

        if not np.any(new_img != img):
            return new_img

        img = new_img


class Distance(Enum):
    Cityblock = 0
    Chebyshev = 1


def distance_transform(img, kind=Distance.Cityblock):
    """
    Perform distance transformation
    :param img: binary image
    :param kind: what kind of distance is chosen
    :return: transformed image, un-normalized
    """

    if not is_binary(img):
        logger.error("distance_transform: input should be binary")
        return img

    # find the border
    row_num, col_num = img.shape
    # border = []
    # for row_i, row in enumerate(img):
    #     for col_i, pixel in enumerate(row):
    #         if pixel == 0:
    #             if row_i != 0 and pixel != img[row_i - 1, col_i]:
    #                 border.append((row_i, col_i))
    #
    #             if row_i != 0 and col_i != col_num - 1 and pixel != img[row_i - 1, col_i + 1]:
    #                 border.append((row_i, col_i))
    #
    #             if row_i != row_num - 1 and pixel != img[row_i + 1, col_i]:
    #                 border.append((row_i, col_i))
    #
    #             if row_i != row_num - 1 and col_i != 0 and pixel != img[row_i + 1, col_i - 1]:
    #                 border.append((row_i, col_i))
    #
    #             if col_i != 0 and pixel != img[row_i, col_i - 1]:
    #                 border.append((row_i, col_i))
    #
    #             if col_i != 0 and row_i != 0 and pixel != img[row_i - 1, col_i - 1]:
    #                 border.append((row_i, col_i))
    #
    #             if col_i != col_num - 1 and pixel != img[row_i, col_i + 1]:
    #                 border.append((row_i, col_i))
    #
    #             if col_i != col_num - 1 and row_i != row_num - 1 and pixel != img[row_i + 1, col_i - 1]:
    #                 border.append((row_i, col_i))

    # cal distance for inner pixel
    if kind == Distance.Cityblock:
        temp0 = np.zeros(img.shape)
        temp1 = np.zeros(img.shape)
        temp2 = np.zeros(img.shape)
        temp3 = np.zeros(img.shape)
        for row_i in range(1, row_num):
            for col_i in range(1, col_num):
                if img[row_i, col_i] == 1:
                    temp0[row_i, col_i] = min(temp0[row_i - 1, col_i], temp0[row_i, col_i - 1]) + 1
        for row_i in range(1, row_num):
            for col_i in range(col_num - 2, -1, -1):
                if img[row_i, col_i] == 1:
                    temp1[row_i, col_i] = min(temp1[row_i - 1, col_i], temp1[row_i, col_i + 1]) + 1
        for row_i in range(row_num - 2, -1, -1):
            for col_i in range(1, col_num):
                if img[row_i, col_i] == 1:
                    temp2[row_i, col_i] = min(temp2[row_i + 1, col_i], temp2[row_i, col_i - 1]) + 1
        for row_i in range(row_num - 2, -1, -1):
            for col_i in range(col_num - 2, -1, -1):
                if img[row_i, col_i] == 1:
                    temp3[row_i, col_i] = min(temp3[row_i + 1, col_i], temp3[row_i, col_i + 1]) + 1
        return np.minimum(np.minimum(temp0, temp1), np.minimum(temp2, temp3))
    else:
        logger.error("distance_transform: unsupported distance")
        return img


def watershed(img, levels=256):
    mask = -2
    wshed = 0
    init = -1
    inqe = -3

    def neighbors(height, width, pixel):
        return np.mgrid[
               max(0, pixel[0] - 1):min(height, pixel[0] + 2),
               max(0, pixel[1] - 1):min(width, pixel[1] + 2)].reshape(2, -1).T

    row_num, col_num = img.shape
    result = np.full((row_num, col_num), init, np.int32)
    current_label = 0

    flag = False
    fifo = deque()

    total = row_num * col_num

    reshaped_image = img.reshape(total)
    # [y, x] pairs of pixel coordinates of the flattened image.
    pixels = np.mgrid[0:row_num, 0:col_num].reshape(2, -1).T
    # Coordinates of neighbour pixels for each pixel.
    neighbours = np.array([neighbors(row_num, col_num, p) for p in pixels])
    if len(neighbours.shape) == 3:
        # Case where all pixels have the same number of neighbours.
        neighbours = neighbours.reshape(row_num, col_num, -1, 2)
    else:
        # Case where pixels may have a different number of pixels.
        neighbours = neighbours.reshape(row_num, col_num)

    indices = np.argsort(reshaped_image)
    sorted_image = reshaped_image[indices]
    sorted_pixels = pixels[indices]

    # levels evenly spaced steps fro minimum to maximum.
    levels = np.linspace(sorted_image[0], sorted_image[-1], levels)
    level_indices = []
    current_level = 0

    # Get the indices that deleimit pixels with different values.
    for i in range(total):
        if sorted_image[i] > levels[current_level]:
            # Skip levels until the next highest one is reached.
            while sorted_image[i] > levels[current_level]:
                current_level += 1
            level_indices.append(i)
    level_indices.append(total)

    start_index = 0
    for stop_index in level_indices:
        # Mask all pixels at the current level.
        for p in sorted_pixels[start_index:stop_index]:
            result[p[0], p[1]] = mask
            # Initialize queue with neighbours of existing basins at the current level.
            for q in neighbours[p[0], p[1]]:
                # p == q is ignored here because result[p] < wshed
                if result[q[0], q[1]] >= wshed:
                    result[p[0], p[1]] = inqe
                    fifo.append(p)
                    break

        # Extend basins.
        while fifo:
            p = fifo.popleft()
            # Label p by inspecting neighbours.
            for q in neighbours[p[0], p[1]]:
                # Don't set lab_p in the outer loop because it may change.
                lab_p = result[p[0], p[1]]
                lab_q = result[q[0], q[1]]
                if lab_q > 0:
                    if lab_p == inqe or (lab_p == wshed and flag):
                        result[p[0], p[1]] = lab_q
                    elif lab_p > 0 and lab_p != lab_q:
                        result[p[0], p[1]] = wshed
                        flag = False
                elif lab_q == wshed:
                    if lab_p == inqe:
                        result[p[0], p[1]] = wshed
                        flag = True
                elif lab_q == mask:
                    result[q[0], q[1]] = inqe
                    fifo.append(q)

        # Detect and process new minima at the current level.
        for p in sorted_pixels[start_index:stop_index]:
            # p is inside a new minimum. Create a new label.
            if result[p[0], p[1]] == mask:
                current_label += 1
                fifo.append(p)
                result[p[0], p[1]] = current_label
                while fifo:
                    q = fifo.popleft()
                    for r in neighbours[q[0], q[1]]:
                        if result[r[0], r[1]] == mask:
                            fifo.append(r)
                            result[r[0], r[1]] = current_label

        start_index = stop_index

    return normalize(result == 0)
