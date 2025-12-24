import cv2
import numpy as np
from typing import Optional

from app.services.image_utils import extract_region
from app.preprocessing.general import preprocess_general
from app.preprocessing.code_digit import preprocess_code_image
from wired_table_rec.utils import ImageOrientationCorrector
from app.services.config_manager import ConfigManager


class CodeRecognizer:
    def __init__(self, model, config_manager: ConfigManager):
        self.model = model
        self.config_manager = config_manager
        self.orientation_corrector = ImageOrientationCorrector()

    def recognize(self, image: np.ndarray) -> Optional[str]:
        try:
            code_coords = self.config_manager.get_region_coordinates("code")
            if not code_coords:
                return None
            code_region = extract_region(
                image,
                {"x1": code_coords[0], "y1": code_coords[1], "x2": code_coords[2], "y2": code_coords[3]},
            )

            code_region = self.orientation_corrector(code_region)
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
                digit = self._recognize_digit(contour, cropped)
                if digit is not None:
                    result += digit

            return result if result else None

        except Exception as e:
            print(f"[CodeRecognizer] Ошибка: {e}")
            return None

    def _recognize_digit(self, contour: np.ndarray, image: np.ndarray) -> Optional[str]:
        try:
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y + h, x:x + w]
            input_data = preprocess_code_image(roi)
            pred = self.model.predict(input_data)
            return str(np.argmax(pred))
        except Exception as e:
            print(f"[CodeRecognizer] Ошибка при распознавании цифры: {e}")
            return None


def recognize_code_from_image(image: np.ndarray, config_manager: ConfigManager, model) -> Optional[str]:
    recognizer = CodeRecognizer(model, config_manager)
    return recognizer.recognize(image)
