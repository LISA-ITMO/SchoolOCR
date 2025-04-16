import numpy as np
from ultralytics import YOLO
import cv2


def extract_table_rows(image, model, conf_threshold=0.5, min_y=1500, row_threshold=20, debug=False):
    """
    Анализирует изображение таблицы и возвращает отсортированные строки с координатами клеток

    Параметры:
        image: изображение в формате numpy array (H,W,C)
        model: загруженная модель YOLO
        conf_threshold: порог уверенности (0-1)
        min_y: минимальная Y-координата для фильтрации
        row_threshold: расстояние для группировки в строки (пикселей)
        debug: вывод отладочной информации

    Возвращает:
        list: список строк, где каждая строка - список клеток,
              отсортированных слева направо, в формате [x1,y1,x2,y2]
    """
    # Выполняем предсказание
    results = model(image)

    # Обработка результатов
    for result in results:
        boxes = result.boxes
        if boxes is None:
            if debug:
                print("Не обнаружено клеток таблицы")
            return []

        # Фильтрация клеток
        mask = (boxes.conf >= conf_threshold) & (boxes.xyxy[:, 1] >= min_y) & (boxes.xyxy[:, 1] <= 3300)
        filtered_boxes = boxes[mask]

        if debug:
            print(f"Обнаружено клеток после фильтрации: {len(filtered_boxes)}")

        if len(filtered_boxes) == 0:
            return []

        # Получение координат
        xyxy = filtered_boxes.xyxy.cpu().numpy()
        y_centers = (xyxy[:, 1] + xyxy[:, 3]) / 2

        # Сортировка по Y для группировки по строкам
        sorted_indices = np.argsort(y_centers)
        sorted_boxes = xyxy[sorted_indices]

        # Группировка по строкам
        row_groups = []
        current_row = []
        y_prev = None

        for box in sorted_boxes:
            y_current = (box[1] + box[3]) / 2
            if y_prev is not None and abs(y_current - y_prev) > row_threshold:
                # Сортируем клетки в строке по X координате (слева направо)
                row_groups.append(sorted(current_row, key=lambda b: (b[0] + b[2]) / 2))
                current_row = []
            current_row.append(box.tolist())
            y_prev = y_current

        # Добавляем последнюю строку
        if current_row:
            row_groups.append(sorted(current_row, key=lambda b: (b[0] + b[2]) / 2))

        if debug:
            print(f"\nРезультат обработки:")
            print(f"Всего строк: {len(row_groups)}")
            for i, row in enumerate(row_groups, 1):
                print(f"Строка {i} ({len(row)} клеток):")
                for j, cell in enumerate(row, 1):
                    print(f"  Клетка {j}: [{cell[0]:.0f}, {cell[1]:.0f}, {cell[2]:.0f}, {cell[3]:.0f}]")

        return row_groups

    return []

