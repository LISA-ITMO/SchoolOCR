import tensorflow as tf
import cv2
import re
import json
import os
import numpy as np
from difflib import get_close_matches
import pytesseract
import fitz  # PyMuPDF
from utils.code_recognition import recognize_code
from utils.table_recognition import recognize_table
from ultralytics import YOLO


def load_config(config_path="../config.json"):
    """Загружает конфигурационный файл"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_region(image, coords):
    """Извлекает регион изображения по координатам"""
    x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
    return image[y1:y2, x1:x2]


def recognize_hat(region_img):
    """Распознает текст в шапке документа"""
    text = pytesseract.image_to_string(region_img, lang="rus").strip()
    return text


def parse_hat_text(text):
    """Извлекает предмет, класс и вариант из текста шапки"""
    pattern = re.compile(
        r"\.\s*([^.]*)\s*\.\s*(\d+)\s*[^.]*\.\s*[^.]*\s*([^\d]*)\s*(\d+)",
        re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        subject = match.group(1).lower()
        grade = match.group(2)
        if '&' in grade:
            grade = grade.replace('&', '8')
        variant = match.group(4)
        return subject, grade, variant
    return None, None, None


def find_closest_key(subject, config):
    """Находит наиболее подходящий ключ в конфиге"""
    keys = [key for key in config.keys() if key not in ["regions"]]
    closest_matches = get_close_matches(subject, keys, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None


def is_pdf(file_data):
    """Проверяет, является ли файл PDF"""
    return len(file_data) > 4 and file_data[:4] == b'%PDF'


def pdf_to_image(pdf_data):
    """Конвертирует PDF в изображение с DPI 300"""
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    page = doc.load_page(0)

    # Устанавливаем DPI 300
    zoom = 300 / 72  # 72 - стандартный DPI в PyMuPDF
    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def process_file(file_path):
    """Обрабатывает файл (PDF или изображение)"""
    with open(file_path, 'rb') as f:
        file_data = f.read()

    if is_pdf(file_data):
        print("Обнаружен PDF файл. Конвертируем в изображение с DPI 300...")
        return pdf_to_image(file_data)
    else:
        print("Обнаружено изображение. Обрабатываем...")
        try:
            image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Не удалось загрузить изображение")
            return image
        except Exception as e:
            raise ValueError(f"Неподдерживаемый формат файла: {str(e)}")


def save_table_image(image, original_path):
    """Сохраняет изображение таблицы с правильным расширением"""
    debug_dir = "debug_tables"
    os.makedirs(debug_dir, exist_ok=True)

    # Извлекаем имя файла без расширения
    filename = os.path.splitext(os.path.basename(original_path))[0]
    output_path = os.path.join(debug_dir, f"table_{filename}.png")  # Явно указываем .png

    # Проверяем, успешно ли сохранено изображение
    if not cv2.imwrite(output_path, image):
        raise ValueError(f"Не удалось сохранить изображение по пути: {output_path}")

    print(f"Изображение таблицы сохранено: {output_path}")
    return output_path


def main(file_path, config_path="../config.json"):
    """Основная функция обработки документа"""
    try:
        # Обработка файла
        image = process_file(file_path)

        # Загрузка конфигурации
        config = load_config(config_path)

        # Загрузка моделей
        mnist_model = tf.keras.models.load_model("../mnist_model.keras")
        yolo_model = YOLO("../cell_detect.pt")
        extended_model = tf.keras.models.load_model("../mnist_recognation_extendend.h5")
        print("Модели успешно загружены.")

        # Обработка шапки
        hat_region = extract_region(image, config["regions"]["hat"])
        hat_text = recognize_hat(hat_region)
        print(f"Распознанный текст шапки: {hat_text}")

        cv2.imshow("Шапка", hat_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Парсинг предмета, класса и варианта
        subject, grade, variant = parse_hat_text(hat_text)
        if not subject or not grade:
            hat_region = extract_region(image, config["regions"]["hat_reserve"])
            hat_text = recognize_hat(hat_region)
            subject, grade, variant = parse_hat_text(hat_text)
            cv2.imshow("Шапка", hat_region)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if not subject or not grade:
            print("Не удалось определить предмет, класс и вариант из шапки.")
            return

        # Поиск в конфиге
        key = f"{subject} {grade}"
        print(f"Ищем ключ в конфиге: {key}")

        if key not in config:
            print(f"Ключ '{key}' не найден в конфиге. Поиск наиболее схожего ключа...")
            closest_key = find_closest_key(subject, config)
            if closest_key:
                print(f"Найден наиболее схожий ключ: {closest_key}")
                key = closest_key
            else:
                print("Не удалось найти подходящий ключ в конфиге.")
                return

        # Распознавание кода
        code_region = extract_region(image, config["regions"]["code"])
        code = recognize_code(code_region, mnist_model)
        print(f"Распознанный код участника: {code}")

        cv2.imshow("Код участника", code_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Обработка таблицы
        table_coords = config[key]["table"]
        table_region = extract_region(image, table_coords)

        # Сохранение и распознавание таблицы
        table_path = save_table_image(table_region, file_path)
        recognized_digits = recognize_table(image, extended_model, yolo_model, config[key], debug=True)

        task_dict = {}
        warnings = []
        total_score = 0

        if not recognized_digits:
            print("Не удалось распознать таблицу")
        else:
            task_numbers = config[key].get("task_numbers", "").split()
            low_confidence = []

            for i, (digit, prob) in enumerate(recognized_digits):
                digit = int(digit)
                prob = round(float(prob), 2)

                if i < len(task_numbers):
                    task_name = task_numbers[i]
                    task_dict[task_name] = (digit, prob)

                    if prob < 0.6:
                        low_confidence.append(task_name)

                    if digit not in [10, 11]:  # Исключаем специальные значения
                        total_score += digit

            if low_confidence:
                warnings.append(f"Низкая уверенность в заданиях: {', '.join(low_confidence)}")

        # Вывод результатов
        print("\nРезультаты распознавания:")
        print(f"Предмет: {subject}")
        print(f"Класс: {grade}")
        print(f"Вариант: {variant}")
        print(f"Код участника: {code}")
        print(f"Общий балл: {total_score}")
        print("\nБаллы по заданиям:")
        for task, (digit, prob) in task_dict.items():
            print(f"  {task}: {digit} (уверенность: {prob})")

        if warnings:
            print("\nПредупреждения:")
            for warning in warnings:
                print(f"  - {warning}")

        cv2.imshow("Таблица", table_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        raise  # Пробрасываем исключение дальше для отладки


if __name__ == "__main__":
    main("scans_jpg/БИО 7 кл 2 в 40/БИО 7 кл 2 в 40_page_10.jpg")