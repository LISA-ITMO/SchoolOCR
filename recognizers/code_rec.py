import cv2
import numpy as np
from typing import Optional, Any, List, Tuple
from wired_table_rec.utils import ImageOrientationCorrector
from recognizers.preprocess import MultiPreprocessor


class CodeRecognizer:
    """Recognizes digital codes in images using contour analysis and MNIST model.

    Attributes:
        min_contour_area: Minimum area for a contour to be considered as a digit.
        padding: Padding around the main code region.
        kernel_size: Size of the morphological operation kernel.
        debug: Enable debug output.
    """

    def __init__(
            self,
            min_contour_area: int = 100,
            padding: int = 10,
            kernel_size: Tuple[int, int] = (3, 3),
            debug: bool = True
    ):
        self.min_contour_area = min_contour_area
        self.padding = padding
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.orientation_corrector = ImageOrientationCorrector()
        self.preprocessor = MultiPreprocessor()
        self.debug = debug

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the image for code recognition."""
        image = self.orientation_corrector(image)
        preprocessed = self.preprocessor.preprocess(image, mode="general_ocr")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dilated = cv2.dilate(preprocessed, self.kernel, iterations=1)
        return gray, dilated

    def _extract_main_region(self, gray: np.ndarray, dilated: np.ndarray) -> np.ndarray:
        """Extract the main code region from the image."""
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Apply padding (оригинальная логика добавляет padding внутрь контура)
        x = max(x + self.padding, 0)
        y = max(y + self.padding, 0)
        w = min(w - 2 * self.padding, gray.shape[1] - x)
        h = min(h - 2 * self.padding, gray.shape[0] - y)

        return gray[y:y + h, x:x + w]

    def _find_digit_contours(self, region: np.ndarray) -> List[np.ndarray]:
        """Find and filter digit contours in the code region."""
        _, binary = cv2.threshold(region, 128, 255, cv2.THRESH_BINARY_INV)
        dilated = cv2.dilate(binary, self.kernel, iterations=1)
        eroded = cv2.erode(dilated, self.kernel, iterations=1)

        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])[3:]  # Skip 3 leftmost contours

        return contours

    def recognize(self, image: np.ndarray, model: Any) -> Optional[str]:
        """Recognize a digital code in the image."""
        try:
            gray, dilated = self._preprocess_image(image)
            main_region = self._extract_main_region(gray, dilated)
            digit_contours = self._find_digit_contours(main_region)

            if not digit_contours:
                return None

            result = []
            for contour in digit_contours:
                x, y, w, h = cv2.boundingRect(contour)
                digit_img = main_region[y:y + h, x:x + w]
                input_data = self.preprocessor.preprocess(digit_img, mode="mnist_code")

                if input_data is not None:
                    pred = model.predict(input_data)
                    result.append(str(np.argmax(pred)))

            return ''.join(result) if result else None

        except Exception as e:
            if self.debug:
                print(f"Error during code recognition: {e}")
            return None