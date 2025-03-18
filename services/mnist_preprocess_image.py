import cv2
import numpy as np

def center_image(image, size=(28, 28), digit_size=(20, 20)):
    """
    Центрирует изображение цифры и изменяет его размер до указанного.
    """
    # Находим контуры на изображении
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # Если контуры не найдены, просто изменяем размер изображения
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    # Находим наибольший контур (предполагаем, что это цифра)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # Вырезаем область с цифрой
    digit_roi = image[y:y + h, x:x + w]

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

def preprocess_image(image, output_size=(28, 28), digit_size=(20, 20)):
    """
    Предобрабатывает изображение для распознавания MNIST.
    """
    try:
        # Проверяем, что изображение не пустое
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        # Преобразуем изображение в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Применяем CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Бинаризация изображения
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Морфологическое закрытие для устранения шума
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Центрируем изображение
        centered = center_image(closed, size=output_size, digit_size=digit_size)

        # Нормализуем изображение
        normalized = centered / 255.0

        # Изменяем форму для модели
        reshaped = normalized.reshape(1, output_size[0], output_size[1], 1)

        return reshaped, centered

    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None, None