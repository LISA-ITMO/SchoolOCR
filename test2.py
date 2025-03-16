import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import json
from difflib import get_close_matches
from lineless_table_rec.utils_table_recover import plot_rec_box_with_logic_info
from wired_table_rec import WiredTableRecognition
from wired_table_rec.utils import ImageOrientationCorrector

from mnist_preprocess_image import preprocess_image
import pytesseract


# Загрузка конфига
def load_config(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Извлечение региона из изображения
def extract_region(image, coords):
    x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
    return image[y1:y2, x1:x2]


# Распознавание текста из шапки
def recognize_hat(region_img):
    text = pytesseract.image_to_string(region_img, lang="rus").strip()
    return text


# Определение предмета и класса из текста шапки
def parse_hat_text(text):
    # Регулярное выражение для извлечения предмета и класса
    # Учитываем, что предмет может состоять из двух слов (например, "русский язык")
    pattern = re.compile(r"ВПР\.\s*([А-Яа-я]+\s*[А-Яа-я]*)\s*\.\s*(\d+)\s*класс", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        subject = match.group(1).lower()  # Приводим к нижнему регистру
        grade = match.group(2)
        return subject, grade
    return None, None


# Распознавание кода участника
def recognize_code(region_img):
    code = pytesseract.image_to_string(region_img, config="--psm 6").strip()
    return code if code.isdigit() else None


# Поиск наиболее схожего ключа в конфиге
def find_closest_key(subject, config):
    # Получаем список всех ключей в конфиге
    keys = [key for key in config.keys() if key not in ["regions"]]

    # Ищем наиболее схожий ключ по предмету
    closest_matches = get_close_matches(subject, keys, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None


# Основной скрипт
def main(image_path, config_path="config.json"):
    # Загрузка конфига
    config = load_config(config_path)

    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Изображение по пути {image_path} не найдено.")

    # Извлечение шапки
    hat_region = extract_region(image, config["regions"]["hat"])
    hat_text = recognize_hat(hat_region)
    print(f"Распознанный текст шапки: {hat_text}")

    # Определение предмета и класса
    subject, grade = parse_hat_text(hat_text)
    if not subject or not grade:
        print("Не удалось определить предмет и класс из шапки.")
        return

    # Формируем ключ для поиска в конфиге
    key = f"{subject} {grade}"
    print(f"Ищем ключ в конфиге: {key}")

    # Поиск ключа в конфиге
    if key not in config:
        print(f"Ключ '{key}' не найден в конфиге. Поиск наиболее схожего ключа...")
        closest_key = find_closest_key(subject, config)
        if closest_key:
            print(f"Найден наиболее схожий ключ: {closest_key}")
            key = closest_key
        else:
            print("Не удалось найти подходящий ключ в конфиге.")
            return

    # Извлечение кода участника
    code_region = extract_region(image, config["regions"]["code"])
    code = recognize_code(code_region)
    print(f"Распознанный код участника: {code}")

    # Извлечение таблицы
    table_coords = config[key]["table"]
    table_region = extract_region(image, table_coords)

    # Инициализация движка для распознавания таблиц
    table_engine = WiredTableRecognition()

    # Преобразуем изображение в градации серого
    gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)

    # Применение бинаризации
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Морфологические операции для соединения пунктирных линий
    kernel = np.ones((3, 3), np.uint8)  # Размер ядра можно настроить
    dilated = cv2.dilate(binary, kernel, iterations=2)  # Дилатация для соединения линий

    # Распознавание таблицы
    html, elasp, polygons, logic_points, ocr_res = table_engine(table_region, need_ocr=False)

    if len(logic_points) != config[key]["total_cells"]:
        html, elasp, polygons, logic_points, ocr_res = table_engine(dilated, need_ocr=False)

    if len(logic_points) != config[key]["total_cells"]:
        img_orientation_corrector = ImageOrientationCorrector()
        # Загрузка и коррекция ориентации изображения
        dilated = img_orientation_corrector(dilated)
        html, elasp, polygons, logic_points, ocr_res = table_engine(dilated, need_ocr=False)


    # Визуализация распознанных областей таблицы
    print("Polygons:", polygons)
    print("Logic Points:", logic_points)
    plot_rec_box_with_logic_info(image_path, "./table_rec_box.jpg", logic_points, polygons)

    # Загрузка модели MNIST
    model = tf.keras.models.load_model("mnist_model.keras")
    print("Модель успешно загружена.")

    # Выделение нужных ячеек (вторая строка)
    second_row_cells = []
    for i, logic in enumerate(logic_points):
        if logic[0] == 1 and logic[1] == 1:  # Вторая строка
            second_row_cells.append(polygons[i])

    # Убираем лишние ячейки (если нужно)
    second_row_cells = second_row_cells[1:]

    third_row_cells = []
    for i, logic in enumerate(logic_points):
        if logic[0] == 3 and logic[1] == 3:  # Вторая строка
            third_row_cells.append(polygons[i])

    if len(third_row_cells) != 0:
        third_row_cells = third_row_cells[1:-2]
        second_row_cells.extend(third_row_cells)

    # Массив для хранения распознанных цифр
    recognized_digits = []

    # Создаем график для визуализации
    plt.figure(figsize=(15, 5))

    # Обработка каждой ячейки
    for i, cell in enumerate(second_row_cells):
        x1, y1, x2, y2 = map(int, cell)
        cell_img = table_region[y1:y2, x1:x2]

        # Проверяем, что изображение ячейки не пустое
        if cell_img is None or cell_img.size == 0:
            print(f"Ячейка {i + 1}: Изображение пустое или невалидное.")
            continue

        # Предобработка изображения ячейки
        input_data, processed_img = preprocess_image(cell_img)
        if input_data is None:
            print(f"Ячейка {i + 1}: Не удалось обработать изображение.")
            continue

        # Распознавание цифры
        predictions = model.predict(input_data)
        predicted_digit = np.argmax(predictions)
        predicted_prob = np.max(predictions)

        # Сохранение распознанной цифры
        recognized_digits.append(predicted_digit)

        # Вывод результата
        print(f"Ячейка {i + 1}: Распознана цифра {predicted_digit} с вероятностью {predicted_prob:.4f}")

        # Визуализация оригинальной ячейки
        plt.subplot(2, len(second_row_cells), i + 1)
        plt.imshow(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original {i + 1}")
        plt.axis('off')

        # Визуализация обработанной ячейки
        plt.subplot(2, len(second_row_cells), i + 1 + len(second_row_cells))
        plt.imshow(processed_img, cmap='gray')
        plt.title(f"Processed {i + 1}\nPred: {predicted_digit}\nProb: {predicted_prob:.4f}")
        plt.axis('off')

    # Отображение графиков
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main("help_imgs/img_7.png")