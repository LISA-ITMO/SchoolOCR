import os
import time
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from wired_table_rec import WiredTableRecognition
from wired_table_rec.utils import ImageOrientationCorrector
from wired_table_rec.utils_table_recover import plot_rec_box_with_logic_info
from utils.mnist_preprocess_cell import preprocess_image
from utils.preprocess_general import preprocess_general


def recognize_table(
        image: np.ndarray,
        model: Any,
        config: Dict[str, Any],
        debug: bool = False,
        morph_kernel_size: int = 2,
        morph_iterations: int = 1,
        split_padding: int = 3
) -> Optional[List[Tuple[int, float]]]:
    """
    Распознает таблицу на изображении и возвращает распознанные цифры с вероятностями.

    Параметры:
        image: Изображение с таблицей (BGR)
        model: Модель для распознавания цифр
        config: Конфигурация таблицы (должна содержать 'rows' и 'total_cells')
        debug: Режим отладки с визуализацией
        morph_kernel_size: Размер ядра для морфологических операций
        morph_iterations: Количество итераций морфологических операций
        split_padding: Отступ при разделении таблицы (в пикселях)

    Возвращает:
        Список кортежей (цифра, вероятность) или None при ошибке
    """
    # Инициализация компонентов
    table_engine = WiredTableRecognition()
    orientation_corrector = ImageOrientationCorrector()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (morph_kernel_size, morph_kernel_size))

    def filter_cells(logic_points, polygons):
        """Фильтрация ячеек на основе логики таблицы"""
        second_row = [polygons[i] for i, logic in enumerate(logic_points) if logic[0] == 1 and logic[1] == 1]

        if config["rows"] == 2:
            second_row = second_row[1:]
            third_row = [polygons[i] for i, logic in enumerate(logic_points) if logic[0] == 3 and logic[1] == 3]
            return second_row + third_row[1:-2]
        return second_row[1:-2]

    def try_recognize(img, corrected=False):
        """Попытка распознавания на одном изображении"""
        nonlocal image  # Чтобы можно было обновить оригинал при коррекции

        # Основная попытка
        _, _, polygons, logic_points, _ = table_engine(img, need_ocr=False)
        cells = filter_cells(logic_points, polygons)

        if len(cells) == config["total_cells"]:
            if corrected:
                image = orientation_corrector(image)
            return cells

        # Попытка разделения для 2-строчных таблиц
        if config["rows"] == 2:
            try:
                _, _, polygons, logic_points, _ = table_engine(img, need_ocr=False)
                second_row = [polygons[i] for i, logic in enumerate(logic_points) if logic[0] == 1 or logic[1] == 1]
                split_y = int(max(polygon[3] for polygon in second_row)) + split_padding

                # Обработка частей
                upper_part = img[:split_y, :]
                lower_part = img[split_y:, :]

                if debug:
                    show_image("Upper Part", upper_part)
                    show_image("Lower Part", lower_part)

                _, _, up_poly, up_logic, _ = table_engine(upper_part, need_ocr=False)
                _, _, lw_poly, lw_logic, _ = table_engine(lower_part, need_ocr=False)

                cells = (
                        [up_poly[i] for i, logic in enumerate(up_logic) if logic[0] == 1 and logic[1] == 1][1:] +
                        [lw_poly[i] for i, logic in enumerate(lw_logic) if logic[0] == 1 and logic[1] == 1][1:-2]
                )

                if len(cells) == config["total_cells"]:
                    return cells
            except Exception as e:
                if debug:
                    print(f"Ошибка при разделении таблицы: {e}")

        return None

    def show_image(title, img):
        """Вспомогательная функция отображения"""
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_failed_case(img, logic, polygons):
        """Сохранение информации о неудачном распознавании"""
        os.makedirs("./bad_tables", exist_ok=True)
        timestamp = int(time.time() * 1000)
        img_path = f"./bad_tables/table_image_{timestamp}.jpg"

        cv2.imwrite(img_path, img)
        plot_rec_box_with_logic_info(img_path, f"./bad_tables/table_rec_box_{timestamp}.jpg", logic, polygons)

        if debug:
            print(f"Найдено ячеек: {len(polygons)}, Ожидалось: {config['total_cells']}")

    # Основной процесс распознавания
    processed_images = {
        "Original": image,
        "Preprocessed": preprocess_general(image),
        "Dilated": cv2.dilate(preprocess_general(image), kernel, iterations=morph_iterations),
        "Eroded": cv2.erode(preprocess_general(image), kernel, iterations=morph_iterations)
    }

    filtered_cells = None
    is_corrected = False

    for name, proc_img in processed_images.items():
        if debug:
            print(f"Попытка распознавания: {name}")

        # Попытка без коррекции ориентации
        cells = try_recognize(proc_img)
        if cells:
            filtered_cells = cells
            break

        # Попытка с коррекцией ориентации
        corrected_img = orientation_corrector(proc_img)
        cells = try_recognize(corrected_img, corrected=True)
        if cells:
            filtered_cells = cells
            is_corrected = True
            break

    # Проверка успешности распознавания
    if not filtered_cells or len(filtered_cells) != config["total_cells"]:
        save_failed_case(image, [], [])
        return None

    # Распознавание цифр в ячейках
    results = []
    if debug:
        plt.figure(figsize=(15, 5))

    for i, cell in enumerate(filtered_cells):
        x1, y1, x2, y2 = map(int, cell)
        cell_img = image[y1:y2, x1:x2]

        if cell_img.size == 0:
            print(f"Пустая ячейка {i + 1}")
            continue

        input_data, _ = preprocess_image(cell_img)
        if input_data is None:
            print(f"Ошибка обработки ячейки {i + 1}")
            continue

        pred = model.predict(input_data)
        digit, prob = np.argmax(pred), np.max(pred)
        results.append((digit, prob))

        if debug:
            plt.subplot(2, len(filtered_cells), i + 1)
            plt.imshow(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Original {i + 1}")
            plt.axis('off')

            plt.subplot(2, len(filtered_cells), i + 1 + len(filtered_cells))
            plt.imshow(input_data.reshape(28, 28), cmap='gray')
            plt.title(f"Processed {i + 1}\nPred: {digit}\nProb: {prob:.4f}")
            plt.axis('off')

    if debug:
        plt.tight_layout()
        plt.show()

    return results