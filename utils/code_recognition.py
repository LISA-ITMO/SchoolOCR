import cv2
import numpy as np
from wired_table_rec.utils import ImageOrientationCorrector

from utils.mnist_preprocess_code import preprocess_image
from utils.preprocess_general import preprocess_general

def recognize_code(image, model):
    img_orientation_corrector = ImageOrientationCorrector()
    # Загрузка и коррекция ориентации изображения
    image = img_orientation_corrector(image)
    preprocessed = preprocess_general(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(preprocessed, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    padding = 10
    x, y = x + padding, y + padding
    w, h = w - 2 * padding, h - 2 * padding
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, gray.shape[1] - x), min(h, gray.shape[0] - y)

    # Вырезаем основную область
    cropped = gray[y:y + h, x:x + w]

    # 4. Поиск цифровых контуров
    _, cropped_binary = cv2.threshold(cropped, 128, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(cropped_binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    cropped_contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров по минимальной площади
    min_contour_area = 100  # Минимальная площадь контура (можно настроить)
    cropped_contours = [c for c in cropped_contours if cv2.contourArea(c) > min_contour_area]

    # Сортировка контуров по координате X (слева направо)
    cropped_contours = sorted(cropped_contours, key=lambda c: cv2.boundingRect(c)[0])

    # Удаление трех крайних левых контуров
    cropped_contours = cropped_contours[3:]

    if len(cropped_contours) == 0:
        return None

    # Создаем копию изображения для отрисовки контуров
    contour_image = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    # Отрисовка всех контуров
    cv2.drawContours(contour_image, cropped_contours, -1, (0, 255, 0), 2)  # Зеленый цвет для контуров

    # 5. Распознавание цифр и конкатенация
    result_number = ""



    # Обработка каждой цифры
    for i, contour in enumerate(cropped_contours):
        x_, y_, w_, h_ = cv2.boundingRect(contour)
        digit_roi = cropped[y_:y_ + h_, x_:x_ + w_]

        # Препроцессинг с использованием rec_digit
        input_data = preprocess_image(digit_roi)

        if input_data is not None:
            pred = model.predict(input_data)
            digit = np.argmax(pred)
            result_number += str(digit)

    return result_number