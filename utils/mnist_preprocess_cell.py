import cv2
import numpy as np

def center_image(image, size=(28, 28), digit_size=(20, 20)):
    """
    Центрирует изображение цифры и изменяет его размер до указанного.
    Выбирает контур, ближайший к центру изображения.
    """
    # Находим контуры на изображении
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # Если контуры не найдены, просто изменяем размер изображения
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    # Вычисляем центр изображения
    height, width = image.shape
    image_center = np.array([width / 2, height / 2])

    # Находим контур, ближайший к центру
    min_distance = float('inf')
    best_contour = None

    for contour in contours:
        # Получаем ограничивающий прямоугольник для контура
        x, y, w, h = cv2.boundingRect(contour)
        # Вычисляем центр контура
        contour_center = np.array([x + w / 2, y + h / 2])
        # Вычисляем расстояние до центра изображения
        distance = np.linalg.norm(contour_center - image_center)

        if distance < min_distance:
            min_distance = distance
            best_contour = contour

    # Если не нашли подходящий контур (хотя такого быть не должно)
    if best_contour is None:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    # Получаем ограничивающий прямоугольник для выбранного контура
    x, y, w, h = cv2.boundingRect(best_contour)

    # Вырезаем область с цифрой
    digit_roi = image[y:y + h, x:x + w]

    digit_roi = cv2.erode(
        digit_roi,
        kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=1
    )

    # Применяем дилатацию к вырезанной цифре
    digit_roi = cv2.dilate(
        digit_roi,
        kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
        iterations=2
    )

    # Сохраняем соотношение сторон
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = digit_size[1]
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = digit_size[0]
        new_w = int(new_h * aspect_ratio)

    # Изменяем размер цифры
    resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Создаем пустое изображение и центрируем цифру
    centered_image = np.zeros(size, dtype=np.uint8)
    offset_x = (size[1] - new_w) // 2
    offset_y = (size[0] - new_h) // 2
    centered_image[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_digit

    return centered_image

def preprocess_image(image, output_size=(28, 28), digit_size=(20, 20), crop_pixels=4):
    try:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        if crop_pixels > 0:
            h, w = image.shape[:2]
            if h > 2*crop_pixels and w > 2*crop_pixels:
                image = image[crop_pixels:h-crop_pixels, crop_pixels:w-crop_pixels]
            else:
                print("Предупреждение: изображение слишком маленькое для обрезки")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        dilated = cv2.dilate(
            binary,
            kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1
        )

        centered = center_image(dilated, size=output_size, digit_size=digit_size)

        normalized = centered / 255.0

        reshaped = normalized.reshape(1, output_size[0], output_size[1], 1)

        return reshaped, centered

    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None, None