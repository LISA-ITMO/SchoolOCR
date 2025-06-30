import cv2
import numpy as np
from typing import Optional


class ImagePreprocessor:
    """Preprocesses images for OCR by performing color correction, binarization and inversion.

    Attributes:
        in_black: Input black point for level adjustment (BGR).
        in_white: Input white point for level adjustment (BGR).
        in_gamma: Gamma correction values (BGR).
        out_black: Output black point.
        out_white: Output white point.
    """

    def __init__(
            self,
            in_black: np.ndarray = np.array([170, 170, 170], dtype=np.float32),
            in_white: np.ndarray = np.array([255, 255, 255], dtype=np.float32),
            in_gamma: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32),
            out_black: np.ndarray = np.array([0, 0, 0], dtype=np.float32),
            out_white: np.ndarray = np.array([255, 255, 255], dtype=np.float32)
    ):
        self.in_black = in_black
        self.in_white = in_white
        self.in_gamma = in_gamma
        self.out_black = out_black
        self.out_white = out_white

    def _adjust_levels(self, img: np.ndarray) -> np.ndarray:
        """Apply color level adjustments to the image."""
        color_level = np.clip((img - self.in_black) / (self.in_white - self.in_black), 0, 255)
        color_level = (color_level ** (1 / self.in_gamma)) * (self.out_white - self.out_black) + self.out_black
        return np.clip(color_level, 0, 255).astype(np.uint8)

    def preprocess(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess image for OCR.

        Args:
            img: Input BGR image as numpy array.

        Returns:
            Preprocessed inverted binary image or None if processing fails.
        """
        try:
            # Convert to 8-bit if needed
            img_8bit = cv2.convertScaleAbs(img)

            # Apply color correction
            leveled = self._adjust_levels(img_8bit)

            # Convert to grayscale and binarize
            gray = cv2.cvtColor(leveled, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Invert for OCR
            return cv2.bitwise_not(binary)

        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            return None