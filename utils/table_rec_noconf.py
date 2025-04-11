from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.mnist_preprocess_cell import preprocess_image
from utils.Yolo_cell_rec import extract_table_rows
from utils.preprocess_general import preprocess_general
import pytesseract
import os
import time


def get_cell_width(cell):
    return cell[2] - cell[0]

def filter_cells(table_rows):
    if len(table_rows) == 2:
        return table_rows[0][1:-2], table_rows[1][1:-2]
    elif len(table_rows) == 4:
        if len(table_rows[1]) >= 2:
            first_cell_width = get_cell_width(table_rows[2][0])
            second_cell_width = get_cell_width(table_rows[2][1])

            if first_cell_width - second_cell_width > 30:
                print("deleting")
                return table_rows[0][1:] + table_rows[2][1:-2], table_rows[1][1:] + table_rows[3][1:-2]
            else:
                return table_rows[0][1:] + table_rows[2][:-2], table_rows[1][1:] + table_rows[3][:-2]
        else:
            return table_rows[0][1:] + table_rows[2][1:-2], table_rows[1][1:] + table_rows[3][1:-2]
    elif len(table_rows) == 6:
        return table_rows[1][1:] + table_rows[4][1:-2], table_rows[2][1:] + table_rows[5][1:-2]

    return None, None

def recognize_table_all(
        image: np.ndarray,
        model_digit: Any,
        model_yolo: Any,
        debug: bool = False,
):

    table_rows = extract_table_rows(image, model_yolo)
    # table_rows = [row for row in table_rows if len(row) >= 3]

    filtered_cells_mnist, filtered_cells_tasks = filter_cells(table_rows)
    if not filtered_cells_mnist or not filtered_cells_tasks:
        return None, None


    print(filtered_cells_mnist)
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
        print(f"Найдено клеток {len(filtered_cells_mnist)}, Ожидалось {len(filtered_cells_tasks)}")
        return None, None

    tasks = [str(i) for i in range(1, len(filtered_cells_tasks) + 1)]
    scores = []
    results = []

    if debug:
        plt.tight_layout()
        plt.show()

    output_dir_original = "./debug_cells/original"  # оригинальные вырезанные цифры
    output_dir_processed = "./debug_cells/processed"  # обработанные (под MNIST)

    # Создаём папки, если их нет
    # os.makedirs(output_dir_original, exist_ok=True)
    # os.makedirs(output_dir_processed, exist_ok=True)

    for i, cell in enumerate(filtered_cells_mnist):
        x1, y1, x2, y2 = map(int, cell)
        cell_img = image[y1:y2, x1:x2]

        if cell_img.size == 0:
            print(f"Пустая ячейка {i + 1}")
            continue

        input_data, _ = preprocess_image(cell_img)
        if input_data is None:
            print(f"Ошибка обработки ячейки {i + 1}")
            continue

        pred = model_digit.predict(input_data)
        digit, prob = np.argmax(pred), np.max(pred)
        scores.append((digit, prob))

        timestamp = int(time.time() * 1000)

        # Сохранение оригинального изображения (cell_img)
        # original_filename = os.path.join(output_dir_original, f"cell_{timestamp}_orig.jpg")
        # cv2.imwrite(original_filename, cell_img)
        #
        # # Сохранение обработанного изображения (input_data)
        # processed_img = (input_data.squeeze() * 255).astype(np.uint8)  # (28, 28) в [0, 255]
        # processed_filename = os.path.join(output_dir_processed, f"cell_{timestamp}_processed.png")
        # cv2.imwrite(processed_filename, processed_img)

        if debug:
            plt.subplot(2, len(filtered_cells_mnist), i + 1)
            plt.imshow(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Original {i + 1}")
            plt.axis('off')

            plt.subplot(2, len(filtered_cells_mnist), i + 1 + len(filtered_cells_mnist))
            plt.imshow(input_data.reshape(28, 28), cmap='gray')
            plt.title(f"Processed {i + 1}\nPred: {digit}\nProb: {prob:.4f}")
            plt.axis('off')

    if debug:
        plt.tight_layout()
        plt.show()

    results.append(tasks)
    results.append(scores)
    return tasks, scores
