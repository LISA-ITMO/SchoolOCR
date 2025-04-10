from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.mnist_preprocess_cell import preprocess_image
from utils.Yolo_cell_rec import extract_table_rows
import pytesseract


def recognize_table_all(
        image: np.ndarray,
        model_digit: Any,
        model_yolo: Any,
        debug: bool = False,
):

    table_rows = extract_table_rows(image, model_yolo)

    if len(table_rows) == 2:
        filtered_cells_tasks = table_rows[0][1:-2]
        filtered_cells_mnist = table_rows[1][1:-2]
    elif len(table_rows) == 4:
        filtered_cells_tasks = table_rows[0][1:] + table_rows[2][1:-2]
        filtered_cells_mnist = table_rows[1][1:] + table_rows[3][1:-2]
    else:
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
        print(f"Найдено клеток {len(filtered_cells_mnist)}, Ожидалось {len(filtered_cells_tasks)}")
        return None, None

    tasks = [str(i) for i in range(1, len(filtered_cells_tasks) + 1)]
    scores = []
    results = []

    # for i, cell in enumerate(filtered_cells_tasks):
    #     x1, y1, x2, y2 = map(int, cell)
    #     cell_img = image[y1:y2, x1:x2]
    #
    #     if cell_img.size == 0:
    #         print(f"Пустая ячейка {i + 1}")
    #         continue
    #
    #     digit = pytesseract.image_to_string(cell_img, lang='rus').strip()
    #     tasks.append(digit)

    if debug:
        plt.tight_layout()
        plt.show()

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
