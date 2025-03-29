import tensorflow as tf
import cv2
import re
import json
import os
import numpy as np
from difflib import get_close_matches
import pytesseract
from pdf2image import convert_from_bytes
from services.code_recognition import recognize_code
from services.table_recognition import recognize_table


def load_config(config_path="../config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_region(image, coords):
    x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
    return image[y1:y2, x1:x2]


def recognize_hat(region_img):
    text = pytesseract.image_to_string(region_img, lang="rus").strip()
    return text


def parse_hat_text(text):
    pattern = re.compile(r"ВПР\.\s*([А-Яа-я]+\s*[А-Яа-я]*)\s*\.\s*(\d+)\s*класс", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        subject = match.group(1).lower()
        grade = match.group(2)
        return subject, grade
    return None, None


def find_closest_key(subject, config):
    keys = [key for key in config.keys() if key not in ["regions"]]
    closest_matches = get_close_matches(subject, keys, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None


def is_pdf(file_data):
    # Проверяем сигнатуру PDF файла (первые 4 байта должны быть '%PDF')
    return len(file_data) > 4 and file_data[:4] == b'%PDF'


def process_file(file_path):
    # Читаем файл как бинарные данные
    with open(file_path, 'rb') as f:
        file_data = f.read()

    # Определяем тип файла по сигнатуре
    if is_pdf(file_data):
        print("Обнаружен PDF файл. Конвертируем первую страницу в изображение...")
        images = convert_from_bytes(file_data, first_page=1, last_page=1)
        image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    else:
        # Пробуем загрузить как изображение
        print("Обнаружено изображение. Обрабатываем...")
        try:
            image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Не удалось загрузить изображение")
        except Exception as e:
            raise ValueError(f"Неподдерживаемый формат файла: {str(e)}")

    return image


def save_table_image(image, original_path):
    """Сохраняет изображение таблицы в папку debug_tables"""
    debug_dir = "debug_tables"
    os.makedirs(debug_dir, exist_ok=True)
    filename = os.path.basename(original_path)
    output_path = os.path.join(debug_dir, f"table_{filename}")
    cv2.imwrite(output_path, image)
    print(f"Изображение таблицы сохранено: {output_path}")


def main(file_path, config_path="../config.json"):
    try:
        # Обрабатываем файл (PDF или изображение)
        image = process_file(file_path)

        # Загрузка конфига
        config = load_config(config_path)

        # Загрузка моделей MNIST
        mnist_model = tf.keras.models.load_model("../mnist_model.keras")
        extended_model = tf.keras.models.load_model("../mnist_recognation_extendend.h5")
        print("Модели успешно загружены.")

        # Извлечение шапки
        hat_region = extract_region(image, config["regions"]["hat"])
        hat_text = recognize_hat(hat_region)
        print(f"Распознанный текст шапки: {hat_text}")

        cv2.imshow("Шапка", hat_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        code = recognize_code(code_region, mnist_model)
        print(f"Распознанный код участника: {code}")

        cv2.imshow("Код участника", code_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Извлечение таблицы
        table_coords = config[key]["table"]
        table_region = extract_region(image, table_coords)

        # Сохранение таблицы
        save_table_image(table_region, file_path)

        # Распознавание таблицы
        recognized_digits = recognize_table(table_region, extended_model, config[key])
        print(f"Распознанные цифры из таблицы: {recognized_digits}")
        cv2.imshow("Таблица", table_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    main("output_images/page_7.jpg")