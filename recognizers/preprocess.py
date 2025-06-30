import cv2
import numpy as np
import math
from typing import Tuple, Optional, Union
from scipy.ndimage import center_of_mass


class MultiPreprocessor:
    """Класс для различных методов предобработки изображений, включая:
    - MNIST cell preprocessing (с морфологическими операциями и центрированием)
    - MNIST code preprocessing (с центром масс и сдвигом)
    - Общий OCR preprocessing (цветокоррекция и бинаризация)
    """

    def __init__(
            self,
            # Параметры для общего OCR preprocessing
            in_black: np.ndarray = np.array([170, 170, 170], dtype=np.float32),
            in_white: np.ndarray = np.array([255, 255, 255], dtype=np.float32),
            in_gamma: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32),
            out_black: np.ndarray = np.array([0, 0, 0], dtype=np.float32),
            out_white: np.ndarray = np.array([255, 255, 255], dtype=np.float32),
            # Параметры для MNIST cell preprocessing
            mnist_output_size: Tuple[int, int] = (28, 28),
            mnist_digit_size: Tuple[int, int] = (20, 20),
            mnist_crop_pixels: int = 4
    ):
        # Параметры общего OCR preprocessing
        self.in_black = in_black
        self.in_white = in_white
        self.in_gamma = in_gamma
        self.out_black = out_black
        self.out_white = out_white

        # Параметры MNIST preprocessing
        self.mnist_output_size = mnist_output_size
        self.mnist_digit_size = mnist_digit_size
        self.mnist_crop_pixels = mnist_crop_pixels

    def _adjust_levels(self, img: np.ndarray) -> np.ndarray:
        """Вспомогательный метод для цветокоррекции (используется в общем OCR preprocessing)."""
        color_level = np.clip((img - self.in_black) / (self.in_white - self.in_black), 0, 255)
        color_level = (color_level ** (1 / self.in_gamma)) * (self.out_white - self.out_black) + self.out_black
        return np.clip(color_level, 0, 255).astype(np.uint8)

    def preprocess_general_ocr(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Общий preprocessing для OCR (цветокоррекция + бинаризация + инверсия).

        Args:
            img: Входное изображение в формате BGR.

        Returns:
            Инвертированное бинарное изображение или None при ошибке.
        """
        try:
            img_8bit = cv2.convertScaleAbs(img)
            leveled = self._adjust_levels(img_8bit)
            gray = cv2.cvtColor(leveled, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.bitwise_not(binary)
        except Exception as e:
            print(f"General OCR preprocessing failed: {e}")
            return None

    def _center_image_mnist_cell(self, image: np.ndarray) -> np.ndarray:
        """Центрирование изображения цифры для MNIST cell preprocessing."""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cv2.resize(image, self.mnist_output_size, interpolation=cv2.INTER_AREA)

        height, width = image.shape
        image_center = np.array([width / 2, height / 2])
        best_contour = min(
            contours,
            key=lambda c: np.linalg.norm(
                np.array(cv2.boundingRect(c)[:2]) + np.array(cv2.boundingRect(c)[2:]) / 2 - image_center
            )
        )

        x, y, w, h = cv2.boundingRect(best_contour)
        digit_roi = image[y:y + h, x:x + w]

        # Морфологические операции
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        digit_roi = cv2.dilate(
            cv2.erode(digit_roi, kernel, iterations=1),
            cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
            iterations=2
        )

        # Изменение размера с сохранением пропорций
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w = self.mnist_digit_size[1]
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = self.mnist_digit_size[0]
            new_w = int(new_h * aspect_ratio)

        resized = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Центрирование
        centered = np.zeros(self.mnist_output_size, dtype=np.uint8)
        offset_x = (self.mnist_output_size[1] - new_w) // 2
        offset_y = (self.mnist_output_size[0] - new_h) // 2
        centered[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        return centered

    def preprocess_mnist_cell(
            self,
            image: np.ndarray,
            return_centered: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional[np.ndarray]]:
        """Preprocessing для MNIST cell (адаптация к формату MNIST с морфологическими операциями).

        Args:
            image: Входное изображение в формате BGR.
            return_centered: Если True, возвращает tuple (нормализованное, центрированное изображение).

        Returns:
            Нормализованное изображение для модели или tuple с центрированным изображением.
        """
        try:
            if image is None or image.size == 0:
                raise ValueError("Input image is empty or invalid.")

            # Обрезка краев
            if self.mnist_crop_pixels > 0 and all(dim > 2 * self.mnist_crop_pixels for dim in image.shape[:2]):
                image = image[self.mnist_crop_pixels:-self.mnist_crop_pixels,
                        self.mnist_crop_pixels:-self.mnist_crop_pixels]

            # Улучшение контраста и бинаризация
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dilated = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

            # Центрирование и нормализация
            centered = self._center_image_mnist_cell(dilated)
            normalized = centered / 255.0
            reshaped = normalized.reshape(1, *self.mnist_output_size, 1)

            return (reshaped, centered) if return_centered else reshaped

        except Exception as e:
            print(f"MNIST cell preprocessing error: {e}")
            return (None, None) if return_centered else None

    @staticmethod
    def _get_best_shift(img: np.ndarray) -> Tuple[int, int]:
        """Вычисление оптимального сдвига для центрирования по центру масс."""
        cy, cx = center_of_mass(img)
        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)
        return shiftx, shifty

    @staticmethod
    def _shift(img: np.ndarray, sx: int, sy: int) -> np.ndarray:
        """Сдвиг изображения."""
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted

    def preprocess_mnist_code(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Preprocessing для MNIST code (центрирование по центру масс).

        Args:
            img: Входное изображение в формате BGR или grayscale.

        Returns:
            Нормализованное изображение для модели MNIST или None при ошибке.
        """
        try:
            # If image is already grayscale (single channel), use it directly
            if len(img.shape) == 2:
                gray = 255 - img  # Invert if needed
            else:
                # Convert from BGR to grayscale if it's a color image
                gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Rest of the code remains the same...
            while np.sum(gray[0]) == 0:
                gray = gray[1:]
            while np.sum(gray[:, 0]) == 0:
                gray = np.delete(gray, 0, 1)
            while np.sum(gray[-1]) == 0:
                gray = gray[:-1]
            while np.sum(gray[:, -1]) == 0:
                gray = np.delete(gray, -1, 1)

            rows, cols = gray.shape
            if rows > cols:
                factor = 20.0 / rows
                rows = 20
                cols = int(round(cols * factor))
                gray = cv2.resize(gray, (cols, rows))
            else:
                factor = 20.0 / cols
                cols = 20
                rows = int(round(rows * factor))
                gray = cv2.resize(gray, (cols, rows))

            left_pad = int(math.floor((28 - cols) / 2.0))
            right_pad = int(math.ceil((28 - cols) / 2.0))
            top_pad = int(math.floor((28 - rows) / 2.0))
            bottom_pad = int(math.ceil((28 - rows) / 2.0))

            gray = np.pad(gray, ((top_pad, bottom_pad), (left_pad, right_pad)), 'constant')

            shiftx, shifty = self._get_best_shift(gray)
            shifted = self._shift(gray, shiftx, shifty)
            gray = shifted

            img = gray / 255.0
            return np.array(img).reshape(-1, 28, 28, 1)

        except Exception as e:
            print(f"MNIST code preprocessing error: {e}")
            return None

    def preprocess(
            self,
            img: np.ndarray,
            mode: str = "mnist_cell",
            **kwargs
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Унифицированный интерфейс для всех методов предобработки.

        Args:
            img: Входное изображение в формате BGR.
            mode: Режим предобработки ("mnist_cell", "mnist_code", "general_ocr").
            **kwargs: Дополнительные аргументы для конкретного метода.

        Returns:
            Результат предобработки в зависимости от выбранного режима.
        """
        if mode == "mnist_cell":
            return self.preprocess_mnist_cell(img, **kwargs)
        elif mode == "mnist_code":
            return self.preprocess_mnist_code(img)
        elif mode == "general_ocr":
            return self.preprocess_general_ocr(img)
        else:
            raise ValueError(f"Unknown preprocessing mode: {mode}")