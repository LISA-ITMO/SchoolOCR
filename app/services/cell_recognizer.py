import numpy as np
from typing import Optional, Dict
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from app.preprocessing.cell_digit import preprocess_cell_image
from app.ml.loader import get_extended_model
from app.ollama_interaction.recognize_digit import classify_image_api
from app.services.recognize_logger import RecognizeLogger


@dataclass
class CellResult:
    """Результат распознавания ячейки"""

    digit: Optional[int] = None
    prob: float = 0.0
    prob_all: Dict[str, float] = None
    llm_digit: Optional[str] = None
    llm_used: bool = False
    task_name: Optional[str] = None


class CellRecognizer:
    def __init__(self, debug: bool = False, use_llm: bool = False):
        self.model = get_extended_model()
        self.debug = debug
        self.use_llm = use_llm
        self.cell_images = []
        self.processed_images = []
        self.digits = []
        self.probabilities = []
        self.tasks = []

    def recognize_cell(self, cell_img: np.ndarray, task_name: str) -> CellResult:
        result = CellResult(task_name=task_name)

        if cell_img.size == 0:
            print(f"Ячейка {task_name}: пустая область, пропуск.")
            return result

        input_data, processed_img = preprocess_cell_image(cell_img)
        if input_data is None:
            print(f"Ячейка {task_name}: не удалось обработать изображение.")
            return result

        pred = self.model.predict(input_data)
        model_digit = int(np.argmax(pred))
        model_prob = float(np.max(pred))
        prob_all = {str(i): round(float(p), 2) for i, p in enumerate(pred.reshape(-1))}

        full_detail = {
            str(i): round(float(p), 100) for i, p in enumerate(pred.reshape(-1))
        }

        try:
            recognize_logger = RecognizeLogger()
            recognize_logger.log(
                initial_image=cell_img,
                preprocessed_image=processed_img,
                recognation_result=model_digit,
                recognation_accuracy=model_prob * 100,
                recognation_detail=full_detail,
            )
        except Exception as e:
            print("Ошибка логирования:", e)

        result.digit = model_digit
        result.prob = model_prob
        result.prob_all = prob_all

        if self.use_llm and model_prob < 0.6:
            try:
                success, buffer = cv2.imencode(".png", cell_img)
                if success:
                    llm_result = classify_image_api(buffer.tobytes())

                    if llm_result != "unknown" and llm_result is not None:
                        result.llm_used = True
                        result.llm_digit = llm_result

                        llm_to_digit = {
                            "0": 0,
                            "1": 1,
                            "2": 2,
                            "3": 3,
                            "4": 4,
                            "5": 5,
                            "6": 6,
                            "7": 7,
                            "8": 8,
                            "9": 9,
                            "x": 11,
                            "X": 11,
                        }

                        if llm_result in llm_to_digit:
                            result.digit = llm_to_digit[llm_result]
                            result.prob = 0.7
            except Exception as e:
                print(f"Ошибка LLM для ячейки {task_name}: {e}")

        if self.debug:
            self.cell_images.append(cell_img)
            self.processed_images.append(processed_img)
            self.digits.append(result.digit)
            self.probabilities.append(result.prob)
            self.tasks.append(task_name)

        return result

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
                plt.imshow(self.cell_images[i], cmap="gray")
            else:
                plt.imshow(cv2.cvtColor(self.cell_images[i], cv2.COLOR_BGR2RGB))
            plt.title(f"Original {self.tasks[i]}")
            plt.axis("off")

            # Предобработанное изображение
            plt.subplot(3, num_cells, num_cells + i + 1)
            plt.imshow(self.processed_images[i], cmap="gray")

            digit_label = (
                "-"
                if self.digits[i] == 10
                else ("x" if self.digits[i] == 11 else str(self.digits[i]))
            )
            plt.title(
                f"Processed {self.tasks[i]}\nPred: {digit_label}\nProb: {self.probabilities[i]:.4f}"
            )
            plt.axis("off")

            # Вероятности всех классов
            plt.subplot(3, num_cells, 2 * num_cells + i + 1)
            classes = list(range(12))
            # Здесь можно добавить реальные вероятности всех классов
            plt.bar(classes, [0.1] * 12)  # Заглушка
            plt.title(f"Probabilities {self.tasks[i]}")
            plt.xlabel("Digit")
            plt.ylabel("Probability")
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
