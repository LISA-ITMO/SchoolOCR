import numpy as np
from typing import Tuple, Optional, Dict, Any
import cv2
import matplotlib.pyplot as plt
from app.preprocessing.cell_digit import preprocess_cell_image
from app.ml.loader import get_extended_model


class CellRecognizer:
    def __init__(self, debug: bool = False):
        self.model = get_extended_model()
        self.debug = debug
        self.cell_images = []
        self.processed_images = []
        self.digits = []
        self.probabilities = []
        self.tasks = []

    def recognize_cell(self, cell_img: np.ndarray, task_name: str) -> Tuple[Optional[int], float, Dict[str, float]]:
        if cell_img.size == 0:
            print(f"Ячейка {task_name}: пустая область, пропуск.")
            return None, 0.0, {}

        input_data, processed_img = preprocess_cell_image(cell_img)
        if input_data is None:
            print(f"Ячейка {task_name}: не удалось обработать изображение.")
            return None, 0.0, {}

        pred = self.model.predict(input_data)
        digit = int(np.argmax(pred))
        prob = float(np.max(pred))
        prob_all = {str(i): round(float(p), 2) for i, p in enumerate(pred.reshape(-1))}

        print(f"Ячейка {task_name}: распознана цифра {digit} с вероятностью {prob:.4f}")

        if self.debug:
            self.cell_images.append(cell_img)
            self.processed_images.append(processed_img)
            self.digits.append(digit)
            self.probabilities.append(prob)
            self.tasks.append(task_name)

        return digit, prob, prob_all

    def recognize_multiple_cells(self, cell_images: list, task_names: list) -> list:
        results = []
        for cell_img, task_name in zip(cell_images, task_names):
            result = self.recognize_cell(cell_img, task_name)
            results.append(result)
        return results

    def show_debug_plot(self):
        if not self.debug or not self.cell_images:
            return

        num_cells = len(self.cell_images)
        fig = plt.figure(figsize=(4 * num_cells, 12))

        for i in range(num_cells):
            # Оригинальное изображение
            plt.subplot(3, num_cells, i + 1)
            if len(self.cell_images[i].shape) == 2:
                plt.imshow(self.cell_images[i], cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(self.cell_images[i], cv2.COLOR_BGR2RGB))
            plt.title(f"Original {self.tasks[i]}")
            plt.axis('off')

            # Предобработанное изображение
            plt.subplot(3, num_cells, num_cells + i + 1)
            plt.imshow(self.processed_images[i], cmap='gray')

            digit_label = '-' if self.digits[i] == 10 else ('x' if self.digits[i] == 11 else str(self.digits[i]))
            plt.title(f"Processed {self.tasks[i]}\nPred: {digit_label}\nProb: {self.probabilities[i]:.4f}")
            plt.axis('off')

            # Вероятности всех классов
            plt.subplot(3, num_cells, 2 * num_cells + i + 1)
            classes = list(range(12))
            # Здесь можно добавить реальные вероятности всех классов
            plt.bar(classes, [0.1] * 12)  # Заглушка
            plt.title(f"Probabilities {self.tasks[i]}")
            plt.xlabel('Digit')
            plt.ylabel('Probability')
            plt.xticks(classes)

        plt.tight_layout()
        plt.show()

    def reset_debug_data(self):
        """Очищает debug данные"""
        self.cell_images.clear()
        self.processed_images.clear()
        self.digits.clear()
        self.probabilities.clear()
        self.tasks.clear()