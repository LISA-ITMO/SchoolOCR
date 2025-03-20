import os
import time

import cv2
import numpy as np
from wired_table_rec import WiredTableRecognition
from wired_table_rec.utils import ImageOrientationCorrector
from wired_table_rec.utils_table_recover import plot_rec_box_with_logic_info

from services.mnist_preprocess_cell import preprocess_image
from services.preprocess_general import preprocess_general


def filter_cells_by_logic(logic_points, polygons, row_config):
    # Фильтрация ячеек второй строки
    second_row_cells = [polygons[i] for i, logic in enumerate(logic_points) if logic[0] == 1 and logic[1] == 1]

    # Убираем лишние ячейки в зависимости от конфигурации
    if row_config["rows"] == 2:
        second_row_cells = second_row_cells[1:]  # Удаляем первую ячейку второй строки

        # Фильтрация ячеек третьей строки
        third_row_cells = [polygons[i] for i, logic in enumerate(logic_points) if logic[0] == 3 and logic[1] == 3]
        third_row_cells = third_row_cells[1:-2]  # Удаляем первую и последние две ячейки третьей строки

        # Объединяем ячейки второй и третьей строк
        second_row_cells.extend(third_row_cells)
    else:
        second_row_cells = second_row_cells[1:-2]  # Удаляем первую и последние две ячейки второй строки

    return second_row_cells


def recognize_table(image, model, config):
    """
    Распознает таблицу на изображении и возвращает распознанные цифры с вероятностями.

    :param image: Изображение с таблицей.
    :param model: Модель для распознавания цифр.
    :param config: Конфигурация с параметрами таблицы.
    :return: Список кортежей (цифра, вероятность).
    """
    # Инициализация движка для распознавания таблиц
    table_engine = WiredTableRecognition()

    preprocessed = preprocess_general(image)

    # Морфологические операции для соединения пунктирных линий
    kernel = np.ones((3, 3), np.uint8)  # Размер ядра можно настроить
    dilated = cv2.dilate(preprocessed, kernel, iterations=2)  # Дилатация для соединения линий

    # Распознавание таблицы
    html, elasp, polygons, logic_points, ocr_res = table_engine(image, need_ocr=False)

    # Выделение нужных ячеек с использованием отдельной функции
    filtered_cells = filter_cells_by_logic(logic_points, polygons, config)

    # Если количество ячеек всё ещё не совпадает, пробуем другие методы
    if len(filtered_cells) != config["total_cells"]:
        html, elasp, polygons, logic_points, ocr_res = table_engine(preprocessed, need_ocr=False)
        filtered_cells = filter_cells_by_logic(logic_points, polygons, config)

    if len(filtered_cells) != config["total_cells"]:
        html, elasp, polygons, logic_points, ocr_res = table_engine(dilated, need_ocr=False)
        filtered_cells = filter_cells_by_logic(logic_points, polygons, config)

    if len(filtered_cells) != config["total_cells"]:
        img_orientation_corrector = ImageOrientationCorrector()
        # Загрузка и коррекция ориентации изображения
        dilated = img_orientation_corrector(dilated)
        html, elasp, polygons, logic_points, ocr_res = table_engine(dilated, need_ocr=False)
        filtered_cells = filter_cells_by_logic(logic_points, polygons, config)

        # Если количество ячеек не совпадает и таблица имеет 2 строки
    if len(filtered_cells) != config["total_cells"] and config["rows"] == 2:
        # Находим нижнюю координату всех полигонов второй строки
        _, _, polygons, logic_points, _ = table_engine(preprocessed, need_ocr=False)
        second_row_cells = [polygons[i] for i, logic in enumerate(logic_points) if logic[0] == 1 or logic[1] == 1]
        bottom_coord = max([polygon[3] for polygon in second_row_cells])  # polygon[3] — это y2 (нижняя координата)
        # print(second_row_cells)
        # print(bottom_coord)

        # Делаем отступ в 3 пикселя и обрезаем изображение на две части
        split_y = int(bottom_coord) + 3
        upper_part = preprocessed[:split_y, :]  # Верхняя часть изображения
        lower_part = preprocessed[split_y:, :]  # Нижняя часть изображения

        # cv2.imshow("upper", upper_part)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # cv2.imshow("lower", lower_part)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Обрабатываем каждую часть отдельно
        _, _, polygons_upper, logic_points_upper, _ = table_engine(upper_part, need_ocr=False)
        _, _, polygons_lower, logic_points_lower, _ = table_engine(lower_part, need_ocr=False)

        # Фильтруем ячейки для каждой части
        filtered_cells_upper = [polygons_upper[i] for i, logic in enumerate(logic_points_upper) if
                                logic[0] == 1 and logic[1] == 1]
        filtered_cells_lower = [polygons_lower[i] for i, logic in enumerate(logic_points_lower) if
                                logic[0] == 1 and logic[1] == 1]

        # Соединяем результаты
        filtered_cells = filtered_cells_upper[1:] + filtered_cells_lower[1:-2]

    if len(filtered_cells) != config["total_cells"]:
        os.makedirs("./bad_tables", exist_ok=True)

        # Генерируем уникальное имя файла с использованием временной метки
        timestamp = int(time.time() * 1000)  # Текущее время в миллисекундах
        output_image_path = f"./bad_tables/table_image_{timestamp}.jpg"

        # Сохраняем текущее изображение
        cv2.imwrite(output_image_path, dilated)

        # Передаем путь к сохраненному изображению в функцию plot_rec_box_with_logic_info
        plot_rec_box_with_logic_info(
            output_image_path,
            f"./bad_tables/table_rec_box_{timestamp}.jpg",
            logic_points,
            polygons
        )
        print(f"Количество отфильтрованных ячеек: {len(filtered_cells)}")
        print(f"Ожидаемое количество ячеек: {config['total_cells']}")
        return None

    # Массив для хранения распознанных цифр с вероятностями
    recognized_results = []

    # Обработка каждой ячейки
    for i, cell in enumerate(filtered_cells):
        x1, y1, x2, y2 = map(int, cell)
        cell_img = image[y1:y2, x1:x2]

        # Проверяем, что изображение ячейки не пустое
        if cell_img is None or cell_img.size == 0:
            print(f"Ячейка {i + 1}: Изображение пустое или невалидное.")
            continue

        # Предобработка изображения ячейки
        input_data, _ = preprocess_image(cell_img)
        if input_data is None:
            print(f"Ячейка {i + 1}: Не удалось обработать изображение.")
            continue

        # Распознавание цифры
        predictions = model.predict(input_data)
        predicted_digit = np.argmax(predictions)
        predicted_prob = np.max(predictions)

        # Сохранение распознанной цифры с вероятностью
        recognized_results.append((predicted_digit, predicted_prob))

    return recognized_results