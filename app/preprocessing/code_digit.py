import math
import cv2
import numpy as np
from scipy.ndimage import center_of_mass

def get_best_shift(img):
    cy, cx = center_of_mass(img)
    rows, cols = img.shape
    return int(round(cols / 2 - cx)), int(round(rows / 2 - cy))

def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    return cv2.warpAffine(img, M, (cols, rows))

def preprocess_code_image(img):
    """
    Преобразует ROI с цифрой в 28x28 нормализованный массив под MNIST-стиль.
    """
    gray = 255 - img
    _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Remove black borders
    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)
    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    rows, cols = gray.shape
    factor = 20.0 / max(rows, cols)
    resized = cv2.resize(gray, (int(cols * factor), int(rows * factor)), interpolation=cv2.INTER_AREA)

    # Padding
    cols_padding = (int(math.ceil((28 - resized.shape[1]) / 2)), int(math.floor((28 - resized.shape[1]) / 2)))
    rows_padding = (int(math.ceil((28 - resized.shape[0]) / 2)), int(math.floor((28 - resized.shape[0]) / 2)))
    padded = np.pad(resized, (rows_padding, cols_padding), 'constant')

    sx, sy = get_best_shift(padded)
    shifted = shift(padded, sx, sy)

    img = shifted / 255.0
    return img.reshape(-1, 28, 28, 1)
