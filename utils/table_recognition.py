from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.mnist_preprocess_cell import preprocess_image
from utils.Yolo_cell_rec import extract_table_rows


def recognize_table(
        image: np.ndarray,
        model_digit: Any,
        model_yolo: Any,
        config: Dict[str, Any],
        debug: bool = False,
) -> Optional[List[Tuple[int, float]]]:

    table_rows = extract_table_rows(image, model_yolo)

    filtered_cells = []
    if config["rows"] == 1:
        filtered_cells = table_rows[1][1:-2]
    if config["rows"] == 2:
        filtered_cells = table_rows[1][1:] + table_rows[3][1:-2]

    if len(filtered_cells) != config["total_cells"]:
        i = 0
        while i < len(filtered_cells) - 1:
            current_x = filtered_cells[i][0]
            next_x = filtered_cells[i + 1][0]

            if abs(next_x - current_x) <= 50:
                filtered_cells.pop(i + 1)
            else:
                i += 1

    if len(filtered_cells) != config["total_cells"]:
        print(f"Найдено клеток {len(filtered_cells)}, Ожидалось {config['total_cells']}")
        return None

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

        pred = model_digit.predict(input_data)
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
