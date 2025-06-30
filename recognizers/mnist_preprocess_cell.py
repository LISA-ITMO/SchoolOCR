import cv2
import numpy as np
from typing import Tuple, Optional


def center_image(
        image: np.ndarray,
        size: Tuple[int, int] = (28, 28),
        digit_size: Tuple[int, int] = (20, 20)
) -> np.ndarray:
    """Center the digit image and resize it to specified dimensions.

    Args:
        image: Binary image containing the digit.
        size: Target output size.
        digit_size: Target digit size before centering.

    Returns:
        Centered digit image.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    # Find contour closest to center
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

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    digit_roi = cv2.dilate(
        cv2.erode(digit_roi, kernel, iterations=1),
        cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
        iterations=2
    )

    # Resize preserving aspect ratio
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = digit_size[1]
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = digit_size[0]
        new_w = int(new_h * aspect_ratio)

    resized = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center in output image
    centered = np.zeros(size, dtype=np.uint8)
    offset_x = (size[1] - new_w) // 2
    offset_y = (size[0] - new_h) // 2
    centered[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    return centered


def preprocess_image(
        image: np.ndarray,
        output_size: Tuple[int, int] = (28, 28),
        digit_size: Tuple[int, int] = (20, 20),
        crop_pixels: int = 4
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Preprocess cell image for MNIST model.

    Args:
        image: Input BGR image.
        output_size: Target output size.
        digit_size: Target digit size before centering.
        crop_pixels: Number of pixels to crop from borders.

    Returns:
        Tuple of (preprocessed array for model, centered image) or (None, None) on error.
    """
    try:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        # Crop borders if possible
        if crop_pixels > 0 and all(dim > 2 * crop_pixels for dim in image.shape[:2]):
            image = image[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]

        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        # Binarize
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dilated = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        # Center and normalize
        centered = center_image(dilated, output_size, digit_size)
        normalized = centered / 255.0
        reshaped = normalized.reshape(1, *output_size, 1)

        return reshaped, centered

    except Exception as e:
        print(f"Image processing error: {e}")
        return None, None