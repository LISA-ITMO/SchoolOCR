import cv2
import numpy as np
import tensorflow as tf
from mnist_preprocess_image import preprocess_image
from wired_table_rec.utils import ImageOrientationCorrector

def process_and_recognize_digits(image_path):
    # Загрузка модели MNIST
    model = tf.keras.models.load_model("mnist_model.keras")
    image = cv2.imread(image_path)

    img_orientation_corrector = ImageOrientationCorrector()
    # Загрузка и коррекция ориентации изображения
    image = img_orientation_corrector(image)

    if image is None:
        raise FileNotFoundError(f"Изображение {image_path} не найдено")

    # Преобразуем изображение в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Бинаризация и морфологические операции
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # 3. Поиск контуров и удаление внешнего бокса
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Применяем паддинг 10px
    padding = 10
    x, y = x + padding, y + padding
    w, h = w - 2 * padding, h - 2 * padding
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, gray.shape[1] - x), min(h, gray.shape[0] - y)

    # Вырезаем основную область
    cropped = gray[y:y + h, x:x + w]

    # 4. Поиск цифровых контуров
    _, cropped_binary = cv2.threshold(cropped, 128, 255, cv2.THRESH_BINARY_INV)
    cropped_contours, _ = cv2.findContours(cropped_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортировка и фильтрация контуров (игнорируем 3 левых)
    cropped_contours = sorted(cropped_contours, key=lambda c: cv2.boundingRect(c)[0])[0:]

    # Создаем копию изображения для отрисовки контуров
    contour_image = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    # Отрисовка всех контуров
    cv2.drawContours(contour_image, cropped_contours, -1, (0, 255, 0), 2)  # Зеленый цвет для контуров

    # 5. Распознавание цифр и конкатенация
    result_number = ""
    for contour in cropped_contours:
        x_, y_, w_, h_ = cv2.boundingRect(contour)
        digit_roi = cropped[y_:y_ + h_, x_:x_ + w_]

        # Предобработка для MNIST
        input_data, _ = preprocess_image(
            cv2.cvtColor(digit_roi, cv2.COLOR_GRAY2BGR))  # Конвертируем в 3-канальное

        if input_data is not None:
            pred = model.predict(input_data)
            digit = np.argmax(pred)
            result_number += str(digit)

    # Показываем изображение с контурами
    cv2.imshow("Контуры цифр", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_number


# Использование
if __name__ == "__main__":
    image_path = "cropped_code/page_1.jpg"
    number = process_and_recognize_digits(image_path)
    print(f"Распознанное число: {number}")