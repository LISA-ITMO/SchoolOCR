import cv2
import numpy as np


def preprocess_general(img):
    """
    Предобработка изображения для OCR.
    Возвращает обработанное изображение в формате NumPy (гравировка).
    """
    # Параметры цветокоррекции
    inBlack = np.array([170, 170, 170], dtype=np.float32)
    inWhite = np.array([255, 255, 255], dtype=np.float32)
    inGamma = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    outBlack = np.array([0, 0, 0], dtype=np.float32)
    outWhite = np.array([255, 255, 255], dtype=np.float32)

    # Преобразование в 8-битный формат
    int8 = cv2.convertScaleAbs(img)

    # Коррекция уровней яркости и контраста
    color_level = np.clip((int8 - inBlack) / (inWhite - inBlack), 0, 255)
    color_level = (color_level ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
    color_level = np.clip(color_level, 0, 255).astype(np.uint8)

    # Преобразование в градации серого
    gray_image = cv2.cvtColor(color_level, cv2.COLOR_BGR2GRAY)

    # Бинаризация
    _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Инвертирование
    inverted = cv2.bitwise_not(threshold)

    # Возвращаем обработанное изображение в формате NumPy
    return inverted