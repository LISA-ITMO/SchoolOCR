import cv2
import numpy as np
from typing import Optional

from app.services.image_utils import extract_region
from app.preprocessing.general import preprocess_general
from app.preprocessing.code_digit import preprocess_code_image
from wired_table_rec.utils import ImageOrientationCorrector


def recognize_code_from_image(image: np.ndarray, config: dict, model) -> Optional[str]:
    try:
        code_region = extract_region(image, config["regions"]["code"])

        code_region = ImageOrientationCorrector()(code_region)
        preprocessed = preprocess_general(code_region)

        gray = cv2.cvtColor(code_region, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        binary = cv2.dilate(preprocessed, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        pad = 10
        x, y = max(0, x + pad), max(0, y + pad)
        w, h = max(1, w - 2 * pad), max(1, h - 2 * pad)
        cropped = gray[y:y + h, x:x + w]

        _, binary_crop = cv2.threshold(cropped, 128, 255, cv2.THRESH_BINARY_INV)
        morph = cv2.erode(cv2.dilate(binary_crop, kernel, iterations=1), kernel, iterations=1)

        digit_contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_contours = [c for c in digit_contours if cv2.contourArea(c) > 100]
        digit_contours = sorted(digit_contours, key=lambda c: cv2.boundingRect(c)[0])

        if len(digit_contours) > 6:
            digit_contours = digit_contours[3:]

        if not digit_contours:
            return None

        result = ""
        for contour in digit_contours:
            x_, y_, w_, h_ = cv2.boundingRect(contour)
            roi = cropped[y_:y_ + h_, x_:x_ + w_]
            input_data = preprocess_code_image(roi)
            pred = model.predict(input_data)
            result += str(np.argmax(pred))

        return result

    except Exception as e:
        print(f"[recognize_code_from_image] Ошибка: {e}")
        return None
