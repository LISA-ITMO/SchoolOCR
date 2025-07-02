from typing import Optional, Tuple, List
import numpy as np

from app.services.table_fallback import recognize_table_all
from app.ml.loader import get_extended_model, get_yolo_model, get_yolo_model_extra


def recognize_scores_table(image: np.ndarray) -> Tuple[Optional[dict], int, Optional[List[str]]]:
    model = get_extended_model()
    yolo = get_yolo_model()
    yolo_extra = get_yolo_model_extra()

    task_numbers, recognized_digits = recognize_table_all(image, model, yolo)
    if not recognized_digits:
        task_numbers, recognized_digits = recognize_table_all(image, model, yolo_extra)

    task_dict = {}
    total_score = 0
    low_confidence = []

    if not recognized_digits:
        return None, 0, None

    for i, (digit, prob, pred_obj) in enumerate(recognized_digits):
        digit = int(digit)
        prob = round(float(prob), 2)

        task_name = task_numbers[i] if i < len(task_numbers) else str(i + 1)
        display_digit = '-' if digit == 10 else ('x' if digit == 11 else digit)
        task_dict[task_name] = (display_digit, prob, pred_obj)

        if prob < 0.6:
            low_confidence.append(task_name)

        if digit not in [10, 11]:
            total_score += digit

    return task_dict, total_score, low_confidence or None
