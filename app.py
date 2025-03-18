from fastapi import FastAPI, HTTPException
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

app = FastAPI()

# Загрузка конфига
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
    text = pytesseract.image_to_string(region_img, lang="rus").strip()
    return text

# Определение предмета и класса из текста шапки
def parse_hat_text(text):
    # Регулярное выражение для извлечения предмета, класса и варианта
    pattern = re.compile(r"ВПР\.\s*([А-Яа-я]+\s*[А-Яа-я]*)\s*\.\s*(\d+)\s*класс\.*\s*Вариант\s*(\d+)", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        subject = match.group(1).lower()  # Приводим к нижнему регистру
        grade = match.group(2)
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

# Ручка для обработки изображения
@app.post("/recognize")
def recognize_image(request: ImageRequest):
    try:
        # Декодируем изображение из base64
        image_data = base64.b64decode(request.image_base64)
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Ошибка обработки изображения")

        # Извлечение шапки
        hat_region = extract_region(image, config["regions"]["hat"])
        hat_text = recognize_hat(hat_region)
        print(f"Распознанный текст шапки: {hat_text}")

        # Определение предмета, класса и варианта
        subject, grade, variant = parse_hat_text(hat_text)
        if not subject or not grade or not variant:
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
                raise HTTPException(status_code=400, detail="Не удалось найти подходящий ключ в конфиге")

        # Извлечение кода участника
        code_region = extract_region(image, config["regions"]["code"])
        code = recognize_code(code_region, mnist_model)  # Используем стандартную модель
        print(f"Распознанный код участника: {code}")

        # Извлечение таблицы
        table_coords = config[key]["table"]
        table_region = extract_region(image, table_coords)

        # Распознавание таблицы
        recognized_digits = recognize_table(table_region, extended_model, config[key])  # Используем расширенную модель
        print(f"Распознанные цифры из таблицы: {recognized_digits}")

        # Преобразуем numpy.int64 в стандартные типы Python
        recognized_digits = [int(digit) for digit in recognized_digits]

        # Формируем JSON-ответ
        return {
            "subject": subject,
            "grade": grade,
            "variant": variant,
            "participant_code": code,
            "scores": recognized_digits
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)