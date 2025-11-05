from typing import List, Tuple, Optional, Any
import numpy as np

from app.detection.yolo_cells import extract_table_rows
from app.preprocessing.cell_digit import preprocess_cell_image


def get_cell_width(cell: List[int]) -> int:
    return cell[2] - cell[0]


def filter_cells(table_rows: List[List[List[int]]]) -> Tuple[Optional[List[List[int]]], Optional[List[List[int]]]]:
    """
    Делит строки таблицы на списки ячеек с номерами задач и ячеек с цифрами.
    """
    if len(table_rows) % 2 != 0:
        table_rows = [row for row in table_rows if len(row) > 3]
        if len(table_rows) % 2 != 0:
            return None, None

    if len(table_rows) == 2:
        return table_rows[0][1:-2], table_rows[1][1:-2]

    elif len(table_rows) == 4:
        first_cell_width = get_cell_width(table_rows[2][0])
        second_cell_width = get_cell_width(table_rows[2][1])

        if first_cell_width - second_cell_width > 30:
            return table_rows[0][1:] + table_rows[2][1:-2], table_rows[1][1:] + table_rows[3][1:-2]
        else:
            return table_rows[0][1:] + table_rows[2][:-2], table_rows[1][1:] + table_rows[3][:-2]

    elif len(table_rows) == 6:
        return table_rows[1][1:] + table_rows[4][1:-2], table_rows[2][1:] + table_rows[5][1:-2]

    return None, None


def recognize_table_all(
    image: np.ndarray,
    model_digit: Any,
    model_yolo: Any,
) -> Tuple[Optional[List[str]], Optional[List[Tuple[int, float, dict]]]]:
    """
    Распознаёт таблицу целиком: номера заданий + оценки без использования конфигурации.
    """
    table_rows = extract_table_rows(image, model_yolo)
    filtered_cells_tasks, filtered_cells_mnist = filter_cells(table_rows)

    if not filtered_cells_mnist or not filtered_cells_tasks:
        return None, None

    if len(filtered_cells_mnist) != len(filtered_cells_tasks):
        i = 0
        while i < len(filtered_cells_mnist) - 1:
            current_x = filtered_cells_mnist[i][0]
            next_x = filtered_cells_mnist[i + 1][0]
            if abs(next_x - current_x) <= 50:
                filtered_cells_mnist.pop(i + 1)
            else:
                i += 1

    if len(filtered_cells_mnist) != len(filtered_cells_tasks):
        return None, None

    tasks = [str(i + 1) for i in range(len(filtered_cells_tasks))]
    scores = []

    for i, cell in enumerate(filtered_cells_mnist):
        x1, y1, x2, y2 = map(int, cell)
        cell_img = image[y1:y2, x1:x2]

        if cell_img.size == 0:
            continue

        input_data, _ = preprocess_cell_image(cell_img)
        if input_data is None:
            continue

        pred = model_digit.predict(input_data)
        digit = int(np.argmax(pred))
        prob = float(np.max(pred))
        prob_all = {str(i): round(float(p), 2) for i, p in enumerate(pred.reshape(-1))}

        scores.append((digit, prob, prob_all))

    return tasks, scores
