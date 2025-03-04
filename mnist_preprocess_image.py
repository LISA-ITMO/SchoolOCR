import cv2
import numpy as np


def center_image(image, size=(28, 28), digit_size=(20, 20)):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    digit_roi = image[y:y + h, x:x + w]

    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = digit_size[1]
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = digit_size[0]
        new_w = int(new_h * aspect_ratio)

    resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    centered_image = np.zeros(size, dtype=np.uint8)

    offset_x = (size[1] - new_w) // 2
    offset_y = (size[0] - new_h) // 2

    centered_image[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_digit

    return centered_image


def preprocess_image(image, output_size=(28, 28), digit_size=(20, 20)):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        centered = center_image(closed, size=output_size, digit_size=digit_size)

        normalized = centered / 255.0

        reshaped = normalized.reshape(1, output_size[0], output_size[1], 1)

        return reshaped, centered

    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None, None
