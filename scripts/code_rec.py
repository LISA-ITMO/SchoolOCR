import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from services.mnist_preprocess_code import preprocess_image  # Импортируем новую функцию препроцессинга
from wired_table_rec.utils import ImageOrientationCorrector

def process_and_recognize_digits(image_path):
    # Загрузка модели MNIST
    print("Загрузка модели MNIST...")
    model = tf.keras.models.load_model("../mnist_model.keras")
    print("Модель успешно загружена.")

    # Загрузка изображения
    print(f"Загрузка изображения: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Изображение {image_path} не найдено")

    # Коррекция ориентации изображения
    print("Коррекция ориентации изображения...")
    img_orientation_corrector = ImageOrientationCorrector()
    image = img_orientation_corrector(image)
    print("Ориентация изображения скорректирована.")

    # Преобразуем изображение в градации серого
    print("Преобразование изображения в градации серого...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Изображение в градациях серого", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Бинаризация изображения
    print("Бинаризация изображения...")
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Бинаризованное изображение", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Морфологические операции
    print("Применение морфологических операций...")
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    cv2.imshow("После морфологических операций", dilated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Поиск контуров и удаление внешнего бокса
    print("Поиск контуров...")
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    print(f"Найден внешний контур с координатами: x={x}, y={y}, w={w}, h={h}")

    # Применяем паддинг
    print("Применение паддинга...")
    padding = 10
    x, y = x + padding, y + padding
    w, h = w - 2 * padding, h - 2 * padding
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, gray.shape[1] - x), min(h, gray.shape[0] - y)
    print(f"Координаты после паддинга: x={x}, y={y}, w={w}, h={h}")

    # Вырезаем основную область
    print("Вырезание основной области...")
    cropped = gray[y:y + h, x:x + w]
    cv2.imshow("Основная область", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Поиск цифровых контуров
    print("Поиск цифровых контуров...")
    _, cropped_binary = cv2.threshold(cropped, 128, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(cropped_binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    cropped_contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Найдено контуров: {len(cropped_contours)}")

    # Фильтрация контуров по минимальной площади
    print("Фильтрация контуров по минимальной площади...")
    min_contour_area = 100  # Минимальная площадь контура (можно настроить)
    cropped_contours = [c for c in cropped_contours if cv2.contourArea(c) > min_contour_area]
    print(f"Контуров после фильтрации: {len(cropped_contours)}")

    # Сортировка контуров по координате X (слева направо)
    print("Сортировка контуров...")
    cropped_contours = sorted(cropped_contours, key=lambda c: cv2.boundingRect(c)[0])

    # Удаление трех крайних левых контуров
    print("Удаление трех крайних левых контуров...")
    cropped_contours = cropped_contours[3:]
    print(f"Контуров после удаления: {len(cropped_contours)}")

    # Создаем копию изображения для отрисовки контуров
    print("Отрисовка контуров...")
    contour_image = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, cropped_contours, -1, (0, 255, 0), 2)  # Зеленый цвет для контуров
    cv2.imshow("Контуры цифр", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Распознавание цифр и конкатенация
    print("Распознавание цифр...")
    result_number = ""
    recognized_digits = []
    predicted_probs = []

    # Создаем график для визуализации
    plt.figure(figsize=(15, 10))

    # Отображение исходного изображения с найденными боксами
    plt.subplot(3, 1, 1)
    image_with_boxes = image.copy()
    for contour in cropped_contours:
        x_, y_, w_, h_ = cv2.boundingRect(contour)
        cv2.rectangle(image_with_boxes, (x + x_, y + y_), (x + x_ + w_, y + y_ + h_), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Исходное изображение с найденными боксами")
    plt.axis('off')

    # Обработка каждой цифры
    for i, contour in enumerate(cropped_contours):
        x_, y_, w_, h_ = cv2.boundingRect(contour)
        digit_roi = cropped[y_:y_ + h_, x_:x_ + w_]

        # Препроцессинг с использованием rec_digit
        input_data = preprocess_image(digit_roi)

        if input_data is not None:
            pred = model.predict(input_data)
            digit = np.argmax(pred)
            prob = np.max(pred)
            result_number += str(digit)
            recognized_digits.append(digit)
            predicted_probs.append(prob)

            # Визуализация оригинальной ячейки
            plt.subplot(3, len(cropped_contours), i + 1 + len(cropped_contours))
            plt.imshow(cv2.cvtColor(digit_roi, cv2.COLOR_GRAY2RGB))
            plt.title(f"Original {i + 1}")
            plt.axis('off')

            # Визуализация обработанной ячейки
            plt.subplot(3, len(cropped_contours), i + 1 + 2 * len(cropped_contours))
            plt.imshow(input_data.reshape(28, 28), cmap='gray')
            plt.title(f"Processed {i + 1}\nPred: {digit}\nProb: {prob:.4f}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Распознанное число: {result_number}")
    return result_number


# Использование
if __name__ == "__main__":
    image_path = "help_imgs/img_15.png"
    number = process_and_recognize_digits(image_path)
    print(f"Распознанное число: {number}")