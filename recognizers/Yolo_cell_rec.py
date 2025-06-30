from typing import List, Optional
import numpy as np
from ultralytics import YOLO
import cv2


class YoloRowExtractor:
    """Extracts and organizes table cells into rows using YOLO object detection."""

    def __init__(
            self,
            conf_threshold: float = 0.5,
            min_y: int = 1500,
            max_y: int = 3300,
            row_threshold: int = 20,
            debug: bool = False
    ):
        self.conf_threshold = conf_threshold
        self.min_y = min_y
        self.max_y = max_y
        self.row_threshold = row_threshold
        self.debug = debug

    def _filter_boxes(self, boxes) -> np.ndarray:
        """Filter detected boxes by confidence and Y-coordinate range."""
        mask = (
                (boxes.conf >= self.conf_threshold) &
                (boxes.xyxy[:, 1] >= self.min_y) &
                (boxes.xyxy[:, 1] <= self.max_y)
        )
        return boxes[mask]

    def _group_into_rows(self, boxes: np.ndarray) -> List[List[List[int]]]:
        """Group detected cells into rows based on Y-coordinates."""
        y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
        sorted_indices = np.argsort(y_centers)
        sorted_boxes = boxes[sorted_indices]

        rows = []
        current_row = []
        prev_y = None

        for box in sorted_boxes:
            current_y = (box[1] + box[3]) / 2

            if prev_y is not None and abs(current_y - prev_y) > self.row_threshold:
                rows.append(self._sort_row_cells(current_row))
                current_row = []

            current_row.append(box.tolist())
            prev_y = current_y

        if current_row:
            rows.append(self._sort_row_cells(current_row))

        return rows

    def _sort_row_cells(self, cells: List[List[int]]) -> List[List[int]]:
        """Sort cells in a row by X-coordinate (left to right)."""
        return sorted(cells, key=lambda cell: (cell[0] + cell[2]) / 2)

    def extract_rows(
            self,
            image: np.ndarray,
            model: YOLO
    ) -> List[List[List[int]]]:
        """Extract and organize table cells from image."""
        try:
            results = model(image)
            if not results or not results[0].boxes:
                if self.debug:
                    print("No table cells detected")
                return []

            filtered = self._filter_boxes(results[0].boxes)
            if self.debug:
                print(f"Cells after filtering: {len(filtered)}")

            if not filtered:
                return []

            boxes_xyxy = filtered.xyxy.cpu().numpy()
            rows = self._group_into_rows(boxes_xyxy)

            if self.debug:
                self._print_debug_info(rows)

            return rows

        except Exception as e:
            print(f"Error extracting table rows: {e}")
            return []

    def _print_debug_info(self, rows: List[List[List[int]]]):
        """Print debug information about detected rows and cells."""
        print("\nProcessing results:")
        print(f"Total rows: {len(rows)}")
        for i, row in enumerate(rows, 1):
            print(f"Row {i} ({len(row)} cells):")
            for j, cell in enumerate(row, 1):
                coords = [f"{coord:.0f}" for coord in cell]
                print(f"  Cell {j}: [{', '.join(coords)}]")