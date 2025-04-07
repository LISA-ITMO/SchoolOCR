import cv2
import numpy as np
from typing import List


def extract_table_contours(
        image: np.ndarray,
        num_contours: int = 2,
        padding: int = 3,
        dilation_kernel_size: int = 5,
        dilation_iterations: int = 2,
        preprocess_func=None
) -> List[np.ndarray]:
    # Предварительная обработка
    if preprocess_func is not None:
        processed = preprocess_func(image)
    else:
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            processed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

    # Дилатация для соединения пунктирных линий
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated = cv2.dilate(processed, kernel, iterations=dilation_iterations)

    # Поиск контуров
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортировка контуров по площади (от большего к меньшему)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:num_contours]

    # Вырезание таблиц с отступом
    cropped_tables = []
    for contour in contours_sorted:
        x, y, w, h = cv2.boundingRect(contour)

        # Добавляем отступ с проверкой границ изображения
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, image.shape[1])
        y2 = min(y + h + padding, image.shape[0])

        cropped_table = image[y1:y2, x1:x2]
        cropped_tables.append(cropped_table)

    return cropped_tables