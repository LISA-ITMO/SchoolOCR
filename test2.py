import tensorflow as tf
import cv2
import re
import json
from difflib import get_close_matches
import pytesseract
from services.code_recognition import recognize_code  # Импортируем функцию для распознавания кода
from services.table_recognition import recognize_table  # Импортируем функцию для распознавания таблицы


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
    pattern = re.compile(r"ВПР\.\s*([А-Яа-я]+\s*[А-Яа-я]*)\s*\.\s*(\d+)\s*класс", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        subject = match.group(1).lower()  # Приводим к нижнему регистру
        grade = match.group(2)
        return subject, grade
    return None, None


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

    # Загрузка моделей MNIST
    mnist_model = tf.keras.models.load_model("mnist_model.keras")  # Стандартная модель
    extended_model = tf.keras.models.load_model("mnist_recognation_extendend.h5")  # Расширенная модель
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

    code = recognize_code(code_region, mnist_model)  # Используем стандартную модель
    print(f"Распознанный код участника: {code}")

    cv2.imshow("Код участника", code_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Извлечение таблицы
    table_coords = config[key]["table"]
    table_region = extract_region(image, table_coords)

    # Распознавание таблицы
    recognized_digits = recognize_table(table_region, extended_model, config[key])  # Используем расширенную модель
    print(f"Распознанные цифры из таблицы: {recognized_digits}")
    cv2.imshow("Таблица", table_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("help_imgs/rus7_3.jpg")
