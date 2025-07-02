import cv2
import numpy as np

def center_image(image, size=(28, 28), digit_size=(20, 20)):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    height, width = image.shape
    image_center = np.array([width / 2, height / 2])

    best_contour = min(
        contours,
        key=lambda c: np.linalg.norm(np.array(cv2.boundingRect(c)[:2]) + np.array(cv2.boundingRect(c)[2:]) / 2 - image_center)
    )

    x, y, w, h = cv2.boundingRect(best_contour)
    digit_roi = image[y:y + h, x:x + w]

    digit_roi = cv2.erode(digit_roi, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
    digit_roi = cv2.dilate(digit_roi, cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)), iterations=2)

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

def preprocess_cell_image(image, output_size=(28, 28), digit_size=(20, 20), crop_pixels=4):
    try:
        if image is None or image.size == 0:
            raise ValueError("Пустое изображение")

        if crop_pixels > 0:
            h, w = image.shape[:2]
            if h > 2*crop_pixels and w > 2*crop_pixels:
                image = image[crop_pixels:h-crop_pixels, crop_pixels:w-crop_pixels]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dilated = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        centered = center_image(dilated, size=output_size, digit_size=digit_size)

        normalized = centered / 255.0
        reshaped = normalized.reshape(1, output_size[0], output_size[1], 1)

        return reshaped, centered

    except Exception as e:
        print(f"[preprocess_cell_image] Ошибка: {e}")
        return None, None
