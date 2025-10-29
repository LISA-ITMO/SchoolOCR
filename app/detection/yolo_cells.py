import numpy as np
from ultralytics import YOLO


def extract_table_rows(
    image,
    model: YOLO,
    conf_threshold: float = 0.5,
    min_y: int = 1500,
    row_threshold: int = 20,
    debug: bool = False
) -> list[list[list[int]]]:
    results = model(image)

    for result in results:
        boxes = result.boxes
        if boxes is None:
            if debug:
                print("Ячеек не найдено")
            return []

        # Фильтрация по уверености и координатам Y
        mask = (
            (boxes.conf >= conf_threshold)
            & (boxes.xyxy[:, 1] >= min_y)
            & (boxes.xyxy[:, 1] <= 3300)
        )

        filtered_boxes = boxes[mask]
        if len(filtered_boxes) == 0:
            return []

        xyxy = filtered_boxes.xyxy.cpu().numpy()
        y_centers = (xyxy[:, 1] + xyxy[:, 3]) / 2
        sorted_indices = np.argsort(y_centers)
        sorted_boxes = xyxy[sorted_indices]
        print(sorted_boxes)

        # Группировка в строки по Y
        row_groups = []
        current_row = []
        y_prev = None

        for box in sorted_boxes:
            y_current = (box[1] + box[3]) / 2
            if y_prev is not None and abs(y_current - y_prev) > row_threshold:
                row_groups.append(sorted(current_row, key=lambda b: (b[0] + b[2]) / 2))
                current_row = []
            current_row.append(box.tolist())
            y_prev = y_current

        if current_row:
            row_groups.append(sorted(current_row, key=lambda b: (b[0] + b[2]) / 2))

        if debug:
            print(f"[YOLO] Обнаружено строк: {len(row_groups)}")
            for i, row in enumerate(row_groups, 1):
                print(f"  Строка {i}: {len(row)} клеток")

        return row_groups

    return []
