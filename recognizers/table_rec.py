from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from recognizers.mnist_preprocess_cell import preprocess_image
from recognizers.Yolo_cell_rec import extract_table_rows


class TableRecognizer:
    """Recognizes digits in table cells using YOLO for detection and MNIST for recognition.

    Args:
        debug: Whether to show debug visualizations.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def _filter_cells(self, table_rows: List[List[List[int]]], config: Dict[str, Any]) -> Optional[List[List[int]]]:
        """Filter and adjust table cells based on configuration."""
        if len(table_rows) > 2:
            return None

        if config["rows"] == 1:
            return table_rows[1][1:-2]
        elif config["rows"] == 2:
            return table_rows[1][1:] + table_rows[3][1:-2]
        return None

    def _remove_duplicate_cells(self, cells: List[List[int]]) -> List[List[int]]:
        """Remove adjacent cells that are too close horizontally."""
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
            config: Dict[str, Any]
    ) -> Optional[List[Tuple[int, float]]]:
        """Recognize digits in table cells.

        Args:
            image: Input image.
            model_digit: MNIST digit recognition model.
            model_yolo: YOLO table detection model.
            config: Configuration dictionary.

        Returns:
            List of (digit, confidence) tuples or None if recognition failed.
        """
        table_rows = extract_table_rows(image, model_yolo)
        filtered_cells = self._filter_cells(table_rows, config)

        if not filtered_cells or len(filtered_cells) != config["total_cells"]:
            filtered_cells = self._remove_duplicate_cells(filtered_cells)
            if len(filtered_cells) != config["total_cells"]:
                print(f"Found {len(filtered_cells)} cells, expected {config['total_cells']}")
                return None

        results = []
        if self.debug:
            plt.figure(figsize=(15, 5))

        for i, cell in enumerate(filtered_cells):
            x1, y1, x2, y2 = map(int, cell)
            cell_img = image[y1:y2, x1:x2]

            if cell_img.size == 0:
                print(f"Empty cell {i + 1}")
                continue

            input_data, _ = preprocess_image(cell_img)
            if input_data is None:
                print(f"Cell {i + 1} processing error")
                continue

            pred = model_digit.predict(input_data)
            results.append((np.argmax(pred), np.max(pred)))

            if self.debug:
                self._plot_cell_debug(cell_img, input_data, i, len(filtered_cells))

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