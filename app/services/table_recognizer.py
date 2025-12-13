import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from app.detection.yolo_cells import extract_table_rows
from app.services.cell_recognizer import CellRecognizer, CellResult
from app.ml.loader import get_yolo_model, get_yolo_model_extra


class TableRecognizer:
    def __init__(
        self,
        debug: bool = False,
        use_llm: bool = False,
        confidence_threshold: float = 0.6,
        llm_trigger_threshold: float = 0.6,
    ):
        self.yolo = get_yolo_model()
        self.yolo_extra = get_yolo_model_extra()
        self.confidence_threshold = confidence_threshold
        self.cell_recognizer = CellRecognizer(
            debug=debug,
            use_llm=use_llm,
            llm_trigger_threshold=llm_trigger_threshold,
        )


    def _get_cell_width(self, cell: List[int]) -> int:
        return cell[2] - cell[0]

    def _filter_cells(self, table_rows: List[List[List[int]]]) -> Tuple[
        Optional[List[List[int]]], Optional[List[List[int]]]]:
        if len(table_rows) % 2 != 0:
            table_rows = [row for row in table_rows if len(row) > 3]
            if len(table_rows) % 2 != 0:
                return None, None

        if len(table_rows) == 2:
            return table_rows[0][1:-2], table_rows[1][1:-2]

        elif len(table_rows) == 4:
            first_cell_width = self._get_cell_width(table_rows[2][0])
            second_cell_width = self._get_cell_width(table_rows[2][1])

            if first_cell_width - second_cell_width > 30:
                return table_rows[0][1:] + table_rows[2][1:-2], table_rows[1][1:] + table_rows[3][1:-2]
            else:
                return table_rows[0][1:] + table_rows[2][:-2], table_rows[1][1:] + table_rows[3][:-2]

        elif len(table_rows) == 6:
            return table_rows[1][1:] + table_rows[4][1:-2], table_rows[2][1:] + table_rows[5][1:-2]

        return None, None

    def _extract_cell_images(self, image: np.ndarray, cells: List[List[int]]) -> List[np.ndarray]:
        cell_images = []
        for cell in cells:
            x1, y1, x2, y2 = map(int, cell)
            cell_img = image[y1:y2, x1:x2]
            cell_images.append(cell_img)
        return cell_images

    def _recognize_table_all(self, image: np.ndarray, model_yolo: Any) -> Tuple[
        Optional[List[str]], Optional[List[CellResult]]]:
        table_rows = extract_table_rows(image, model_yolo)
        filtered_cells_tasks, filtered_cells_mnist = self._filter_cells(table_rows)

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
        cell_images = self._extract_cell_images(image, filtered_cells_mnist)

        # Теперь возвращаем список CellResult
        recognition_results = self.cell_recognizer.recognize_multiple_cells(cell_images, tasks)

        return tasks, recognition_results

    def recognize_scores_table(self, image: np.ndarray) -> Tuple[
        Optional[dict], Optional[dict], int, Optional[List[str]]]:
        if self.cell_recognizer.debug:
            self.cell_recognizer.reset_debug_data()

        task_numbers, cell_results = self._recognize_table_all(image, self.yolo)
        if not cell_results:
            task_numbers, cell_results = self._recognize_table_all(image, self.yolo_extra)

        if self.cell_recognizer.debug:
            self.cell_recognizer.show_debug_plot()

        task_dict = {}
        task_dict_prob_details = {}
        total_score = 0
        low_confidence = []

        if not cell_results:
            return None, None, 0, None

        for i, result in enumerate(cell_results):
            digit = result.digit
            prob = round(float(result.prob), 2) if result.prob else 0.0

            task_name = task_numbers[i] if i < len(task_numbers) else str(i + 1)
            display_digit = '-' if digit == 10 else ('x' if digit == 11 else digit)

            task_dict[task_name] = (display_digit, prob)
            task_dict_prob_details[task_name] = result.prob_all if result.prob_all else {}

            if prob < self.confidence_threshold:
                low_confidence.append(task_name)

            if digit not in [10, 11] and digit is not None:
                total_score += digit

        return task_dict, task_dict_prob_details, total_score, low_confidence or None