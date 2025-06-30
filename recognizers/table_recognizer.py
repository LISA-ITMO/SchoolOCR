from typing import List, Tuple, Optional, Dict, Any, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt
from recognizers.preprocess import MultiPreprocessor
from recognizers.Yolo_cell_rec import YoloRowExtractor


class TableRecognizer:
    """Recognizes digits in table cells with or without predefined configuration.

    Combines functionality from TableRecognizer and TableRecognizerNoConfig.
    Supports both configured and automatic table recognition modes.
    """

    def __init__(self, debug: bool = False):
        """
        Args:
            debug: Enable debug visualizations.
        """
        self.debug = debug
        self.preprocessor = MultiPreprocessor()
        self.row_extractor = YoloRowExtractor(debug=debug)

    @staticmethod
    def get_cell_width(cell: List[int]) -> int:
        """Calculate cell width from coordinates."""
        return cell[2] - cell[0]

    def _filter_cells_with_config(self, table_rows: List[List[List[int]]], config: Dict[str, Any]) -> Optional[
        List[List[int]]]:
        """Filter cells using predefined configuration."""
        if len(table_rows) > 2:
            return None

        if config["rows"] == 1:
            return table_rows[1][1:-2]
        elif config["rows"] == 2:
            return table_rows[1][1:] + table_rows[3][1:-2]
        return None

    def _filter_cells_auto(self, table_rows: List[List[List[int]]]) -> Tuple[
        Optional[List[List[int]]], Optional[List[List[int]]]]:
        """Automatically filter and split cells into tasks and digits."""
        if not table_rows:
            return None, None

        # Remove rows with too few cells
        table_rows = [row for row in table_rows if len(row) > 3]
        if len(table_rows) % 2 != 0:
            return None, None

        if len(table_rows) == 2:
            return table_rows[0][1:-2], table_rows[1][1:-2]
        elif len(table_rows) == 4:
            first_width = self.get_cell_width(table_rows[2][0])
            second_width = self.get_cell_width(table_rows[2][1])

            if first_width - second_width > 30:
                return (
                    table_rows[0][1:] + table_rows[2][1:-2],
                    table_rows[1][1:] + table_rows[3][1:-2]
                )
            return (
                table_rows[0][1:] + table_rows[2][:-2],
                table_rows[1][1:] + table_rows[3][:-2]
            )
        elif len(table_rows) == 6:
            return (
                table_rows[1][1:] + table_rows[4][1:-2],
                table_rows[2][1:] + table_rows[5][1:-2]
            )

        return None, None

    def _remove_adjacent_cells(self, cells: List[List[int]]) -> List[List[int]]:
        """Remove cells that are too close to each other."""
        filtered = []
        prev_x = -float('inf')

        for cell in cells:
            current_x = cell[0]
            if abs(current_x - prev_x) > 50:  # Minimum distance between cells
                filtered.append(cell)
                prev_x = current_x

        return filtered

    def recognize(
            self,
            image: np.ndarray,
            model_digit: Any,
            model_yolo: Any,
            config: Optional[Dict[str, Any]] = None
    ) -> Union[
        Optional[List[Tuple[int, float]]],
    ]:
        """Recognize table content with or without configuration.

        Args:
            image: Input image (BGR or RGB numpy array).
            model_digit: Digit recognition model.
            model_yolo: YOLO table detection model.
            config: Configuration dictionary or None for automatic mode.

        Returns:
            - With config: List of (digit, confidence) tuples
            - Without config: Tuple of (task numbers, digit predictions)
            Returns None(s) on failure
        """
        try:
            table_rows = self.row_extractor.extract_rows(image, model_yolo)
            task_cells, digit_cells = self._filter_cells_auto(table_rows)
            if not digit_cells or not task_cells:
                return None, None

            digit_cells = self._remove_adjacent_cells(digit_cells)
            if len(digit_cells) != len(task_cells):
                print(f"Found {len(digit_cells)} digit cells, expected {len(task_cells)}")
                return None, None

            scores = self._process_cells(digit_cells, image, model_digit)

            return scores if scores else None

        except Exception as e:
            print(f"Table recognition error: {e}")
            return None

    def _process_cells(
            self,
            cells: List[List[int]],
            image: np.ndarray,
            model_digit: Any
    ) -> Optional[List[Tuple[int, float]]]:
        """Process cell images through the recognition pipeline."""
        results = []
        if self.debug:
            plt.figure(figsize=(15, 5))

        for i, cell in enumerate(cells):
            x1, y1, x2, y2 = map(int, cell)
            cell_img = image[y1:y2, x1:x2]

            if cell_img.size == 0:
                print(f"Empty cell {i + 1}")
                continue

            input_data = self.preprocessor.preprocess(cell_img, mode="mnist_cell")
            if input_data is None:
                print(f"Cell {i + 1} processing error")
                continue

            pred = model_digit.predict(input_data)
            results.append((np.argmax(pred), np.max(pred)))

            if self.debug:
                self._plot_cell_debug(cell_img, input_data, i, len(cells))

        if self.debug:
            plt.tight_layout()
            plt.show()

        return results if results else None

    def _plot_cell_debug(self, cell_img: np.ndarray, input_data: np.ndarray, idx: int, total: int):
        """Plot debug information for a cell."""
        plt.subplot(2, total, idx + 1)
        plt.imshow(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original {idx + 1}")
        plt.axis('off')

        plt.subplot(2, total, idx + 1 + total)
        plt.imshow(input_data.reshape(28, 28), cmap='gray')
        plt.title(f"Processed {idx + 1}\nPred: {np.argmax(pred)}\nProb: {np.max(pred):.4f}")
        plt.axis('off')