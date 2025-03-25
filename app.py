from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import tensorflow as tf
import json
import re
from difflib import get_close_matches
import pytesseract
from services.code_recognition import recognize_code
from services.table_recognition import recognize_table
from services.preprocess_general import preprocess_general

app = FastAPI()

# Загрузка конфига

API_KEYS_PATH = "api_keys.json"
try:
    with open(API_KEYS_PATH, "r", encoding="utf-8") as f:
        api_keys_config = json.load(f)
    API_KEYS = set(api_keys_config.get("keys", []))
except FileNotFoundError:
    API_KEYS = set()
    print("Файл api_keys.json не найден. API-ключи не загружены.")

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

# Загрузка моделей MNIST
mnist_model = tf.keras.models.load_model("mnist_model.keras")  # Стандартная модель
extended_model = tf.keras.models.load_model("mnist_recognation_extendend.h5")  # Расширенная модель


class ImageRequest(BaseModel):
    image_base64: str


# Извлечение региона из изображения
def extract_region(image, coords):
    x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
    return image[y1:y2, x1:x2]


# Распознавание текста из шапки
def recognize_hat(region_img):
    processed_img = preprocess_general(region_img)

    whitelist = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя.0123456789"
    custom_config = f'-c tessedit_char_whitelist="{whitelist}" --psm 6'

    text = pytesseract.image_to_string(processed_img, lang='rus', config=custom_config).strip()
    return text


# Определение предмета и класса из текста шапки
def parse_hat_text(text):
    # Регулярное выражение для извлечения предмета, класса и варианта
    pattern = re.compile(
        r"\.\s*([^.]*)\s*\.\s*(\d+)\s*[^.]*\.\s*([^.]*)\s*(\d+)",
        re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        subject = match.group(1).lower()
        grade = match.group(2)
        if '&' in grade:
            grade = grade.replace('&', '8')
        variant = match.group(3)
        return subject, grade, variant

    return None, None, None


# Поиск наиболее схожего ключа в конфиге
def find_closest_key(subject, config):
    # Получаем список всех ключей в конфиге
    keys = [key for key in config.keys() if key not in ["regions"]]

    # Ищем наиболее схожий ключ по предмету
    closest_matches = get_close_matches(subject, keys, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None


# Сопоставление распознанных цифр с номерами заданий
def map_digits_to_tasks(recognized_digits, task_numbers):
    # Разделяем task_numbers на отдельные номера заданий
    tasks = task_numbers.split()
    # Создаем словарь для сопоставления
    task_dict = {}
    for i, digit in enumerate(recognized_digits):
        if i < len(tasks):
            task_dict[tasks[i]] = digit
        else:
            task_dict[f"extra_{i}"] = digit  # Если цифр больше, чем заданий
    return task_dict


# Проверка API-ключа
def validate_api_key(api_key: str):
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")


# Ручка для обработки изображения
@app.post("/recognize")
def recognize_image(request: ImageRequest, authorization: str = Header(None)):
    errors = []
    print(1)
    # Проверяем API-ключ
    validate_api_key(authorization)
    print(2)

    try:
        # Декодируем изображение из base64
        image_data = base64.b64decode(request.image_base64)
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            errors.append("Ошибка обработки изображения")
            raise HTTPException(status_code=400, detail="Ошибка обработки изображения")

        # Извлечение шапки
        hat_region = extract_region(image, config["regions"]["hat"])
        hat_text = recognize_hat(hat_region)
        print(f"Распознанный текст шапки: {hat_text}")

        # Определение предмета, класса и варианта
        subject, grade, variant = parse_hat_text(hat_text)
        if not subject or not grade or not variant:
            errors.append("Не удалось определить предмет, класс или вариант из шапки")
            raise HTTPException(status_code=400, detail="Не удалось определить предмет, класс или вариант из шапки")

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
                errors.append("Не удалось найти подходящий ключ в конфиге")
                raise HTTPException(status_code=400, detail="Не удалось найти подходящий ключ в конфиге")

        # Извлечение кода участника
        code_region = extract_region(image, config["regions"]["code"])
        code = recognize_code(code_region, mnist_model)  # Используем стандартную модель
        print(f"Распознанный код участника: {code}")

        if not code:
            errors.append("Не удалось распознать код участника")

        # Извлечение таблицы
        table_coords = config[key]["table"]
        table_region = extract_region(image, table_coords)

        # Распознавание таблицы
        recognized_digits = recognize_table(table_region, extended_model, config[key])

        task_dict = {}
        warnings = []
        total_score = 0

        if not recognized_digits:
            errors.append("Не удалось распознать таблицу")
        else:
            recognized_digits = [(int(digit), round(float(probability), 2))
                                 for digit, probability in recognized_digits]

            task_numbers = config[key].get("task_numbers", "").split()
            low_confidence_tasks = []

            for i, (digit, prob) in enumerate(recognized_digits):
                if i < len(task_numbers):
                    task_name = task_numbers[i]
                    task_dict[task_name] = (digit, prob)

                    # Проверяем вероятность и добавляем номер задания, если < 0.6
                    if prob < 0.6:
                        low_confidence_tasks.append(task_name)

                    # Суммируем баллы, исключая задания 10 и 11
                    if digit not in [10, 11]:
                        total_score += digit
                else:
                    task_dict[f"extra_{i}"] = (digit, prob)

            # Добавляем предупреждение с номерами заданий
            if low_confidence_tasks:
                warnings.append(f"Низкая вероятность распознавания для заданий: {', '.join(low_confidence_tasks)}")

        response = {
            "subject": subject,
            "grade": grade,
            "variant": variant,
            "participant_code": code,
            "total_score": total_score,
            "scores": task_dict,
            "errors": errors if errors else None,
            "warnings": warnings if warnings else None
        }

        return response

    except Exception as e:
        errors.append(str(e))
        raise HTTPException(status_code=500, detail={"errors": errors})


# Запуск сервера
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
