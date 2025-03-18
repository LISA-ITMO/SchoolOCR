import cv2
import numpy as np
from wired_table_rec import WiredTableRecognition
from wired_table_rec.utils import ImageOrientationCorrector

from services.mnist_preprocess_image import preprocess_image


def recognize_table(image, model, config):
    """
    Распознает таблицу на изображении и возвращает распознанные цифры.

    :param image: Изображение с таблицей.
    :param model: Модель для распознавания цифр.
    :param config: Конфигурация с параметрами таблицы.
    :return: Список распознанных цифр.
    """
    # Инициализация движка для распознавания таблиц
    table_engine = WiredTableRecognition()

    # Преобразуем изображение в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение бинаризации
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Морфологические операции для соединения пунктирных линий
    kernel = np.ones((3, 3), np.uint8)  # Размер ядра можно настроить
    dilated = cv2.dilate(binary, kernel, iterations=2)  # Дилатация для соединения линий

    # Распознавание таблицы
    html, elasp, polygons, logic_points, ocr_res = table_engine(image, need_ocr=False)

    if len(logic_points) != config["total_cells"]:
        html, elasp, polygons, logic_points, ocr_res = table_engine(dilated, need_ocr=False)

    if len(logic_points) != config["total_cells"]:
        img_orientation_corrector = ImageOrientationCorrector()
        # Загрузка и коррекция ориентации изображения
        dilated = img_orientation_corrector(dilated)
        html, elasp, polygons, logic_points, ocr_res = table_engine(dilated, need_ocr=False)

    # Выделение нужных ячеек (вторая строка)
    second_row_cells = []
    for i, logic in enumerate(logic_points):
        if logic[0] == 1 and logic[1] == 1:  # Вторая строка
            second_row_cells.append(polygons[i])

    # Убираем лишние ячейки (если нужно)
    if config["rows"] == 2:
        second_row_cells = second_row_cells[1:]

        third_row_cells = []
        for i, logic in enumerate(logic_points):
            if logic[0] == 3 and logic[1] == 3:  # Вторая строка
                third_row_cells.append(polygons[i])
        third_row_cells = third_row_cells[1:-2]
        second_row_cells.extend(third_row_cells)

    else:
        second_row_cells = second_row_cells[1:-2]

    # Массив для хранения распознанных цифр
    recognized_digits = []

    # Обработка каждой ячейки
    for i, cell in enumerate(second_row_cells):
        x1, y1, x2, y2 = map(int, cell)
        cell_img = image[y1:y2, x1:x2]

        # Проверяем, что изображение ячейки не пустое
        if cell_img is None or cell_img.size == 0:
            print(f"Ячейка {i + 1}: Изображение пустое или невалидное.")
            continue

        # Предобработка изображения ячейки
        input_data, processed_img = preprocess_image(cell_img)
        if input_data is None:
            print(f"Ячейка {i + 1}: Не удалось обработать изображение.")
            continue

        # Распознавание цифры
        predictions = model.predict(input_data)
        predicted_digit = np.argmax(predictions)
        predicted_prob = np.max(predictions)

        # Сохранение распознанной цифры
        recognized_digits.append(predicted_digit)

        # Вывод результата
        print(f"Ячейка {i + 1}: Распознана цифра {predicted_digit} с вероятностью {predicted_prob:.4f}")

    return recognized_digits