from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import tensorflow as tf
import json
import re
import fitz  # PyMuPDF
from difflib import get_close_matches
import pytesseract
from services.code_recognition import recognize_code
from services.table_recognition import recognize_table
from services.preprocess_general import preprocess_general

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка конфигурации
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
mnist_model = tf.keras.models.load_model("mnist_model.keras")
extended_model = tf.keras.models.load_model("mnist_recognation_extendend.h5")


class ImageRequest(BaseModel):
    image_base64: str


def is_pdf(file_data):
    """Проверяет, является ли файл PDF"""
    return len(file_data) > 4 and file_data[:4] == b'%PDF'


def pdf_to_image(pdf_data):
    """Конвертирует PDF в изображение с DPI 300"""
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    page = doc.load_page(0)

    # Устанавливаем DPI 300
    zoom = 300 / 72
    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def decode_image(image_data):
    """Обрабатывает изображение или PDF"""
    if is_pdf(image_data):
        print("Обнаружен PDF файл. Конвертируем...")
        return pdf_to_image(image_data)
    else:
        print("Обнаружено изображение. Обрабатываем...")
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Неверный формат изображения")
        return resize_to_target(image)


def resize_to_target(image, target_width=2480, target_height=3505):
    """Изменяет размер изображения до целевого"""
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)


def extract_region(image, coords):
    """Извлекает регион изображения по координатам"""
    x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
    return image[y1:y2, x1:x2]


def recognize_hat(region_img):
    """Распознает текст в шапке документа"""
    processed_img = preprocess_general(region_img)
    whitelist = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя.0123456789"
    custom_config = f'-c tessedit_char_whitelist="{whitelist}" --psm 6'
    text = pytesseract.image_to_string(processed_img, lang='rus', config=custom_config).strip()
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


def validate_api_key(api_key: str):
    """Проверяет валидность API ключа"""
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")


@app.post("/recognize")
def recognize_image(request: ImageRequest, authorization: str = Header(None)):
    """Основной endpoint для распознавания"""
    errors = []
    validate_api_key(authorization)

    try:
        # Декодируем изображение
        image_data = base64.b64decode(request.image_base64)
        image = decode_image(image_data)

        # Извлечение шапки
        hat_region = extract_region(image, config["regions"]["hat"])
        hat_text = recognize_hat(hat_region)
        print(f"Распознанный текст шапки: {hat_text}")

        # Парсинг данных
        subject, grade, variant = parse_hat_text(hat_text)
        if not subject or not grade or not variant:
            raise HTTPException(status_code=400, detail="Не удалось определить предмет, класс или вариант")

        # Поиск конфигурации
        key = f"{subject} {grade}"
        if key not in config:
            closest_key = find_closest_key(subject, config)
            if not closest_key:
                raise HTTPException(status_code=400, detail="Не найдена конфигурация для предмета и класса")
            key = closest_key

        # Распознавание кода
        code_region = extract_region(image, config["regions"]["code"])
        code = None
        try:
            code = recognize_code(code_region, mnist_model)
        except:
            errors.append("Не удалось распознать код участника")

        # Распознавание таблицы
        table_region = extract_region(image, config[key]["table"])
        recognized_digits = recognize_table(table_region, extended_model, config[key])

        # Формирование ответа
        task_dict = {}
        warnings = []
        total_score = 0

        if not recognized_digits:
            errors.append("Не удалось распознать таблицу")
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

                    if digit not in [10, 11]:
                        total_score += digit

            if low_confidence:
                warnings.append(f"Низкая уверенность в заданиях: {', '.join(low_confidence)}")

        return {
            "subject": subject,
            "grade": grade,
            "variant": variant,
            "participant_code": code,
            "total_score": total_score,
            "scores": task_dict,
            "errors": errors if errors else None,
            "warnings": warnings if warnings else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
