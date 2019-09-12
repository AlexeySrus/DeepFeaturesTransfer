import numpy as np
import cv2


def upper_bin(img, threshold):
    res = img.copy()
    res[img > threshold] = 255
    res[img <= threshold] = 0
    return res


def ring_by_np(size):
    res = np.zeros(shape=(size, size), dtype=np.uint8)
    m = size // 2
    for i in range(size):
        for j in range(size):
            if (i - m) ** 2 + (j - m) ** 2 <= m ** 2:
                res[i][j] = 255
    return res


def increase_sharpen(img):
    img_blured = cv2.GaussianBlur(img, (5, 5), 0)
    img_m = cv2.addWeighted(img, 1.5, img_blured, -0.5, 0)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_s = cv2.filter2D(img_m, -1, kernel, borderType=cv2.CV_8U)
    return img_s


def image_preprocessing(img):
    im = cv2.cvtColor(increase_sharpen(img), cv2.COLOR_RGB2GRAY)
    th = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    im = upper_bin(im, th)

    k = ring_by_np(5)
    im = upper_bin(cv2.morphologyEx(im, cv2.MORPH_DILATE, k), 10)

    im = 255 - im

    k = np.ones((15, 15))
    im = upper_bin(cv2.morphologyEx(im, cv2.MORPH_DILATE, k), 10)
    k = np.ones((55, 55))
    im = upper_bin(cv2.morphologyEx(im, cv2.MORPH_ERODE, k), 10)
    k = np.ones((55 - 15, 55 - 15))
    im = upper_bin(cv2.morphologyEx(im, cv2.MORPH_DILATE, k), 10)
    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
