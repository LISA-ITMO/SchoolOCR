import requests
import base64
import json

# === Настройки ===
MODEL = "qwen2.5vl:7b"
PROMPT = (
    "Ты очень внимательный OCR инструмент. Определи, какой символ (0–9 или X) "
    "изображён на картинке. Часто цифры пишут неаккуратно — учти это. "
    "Ответь только этим символом. Можешь отвечать только 0-9 или X"
)
OLLAMA_API = "http://localhost:11434/api/generate"  # твой локальный эндпоинт


def classify_image_api(image_bytes):
    """
    Классификация одного изображения через Ollama HTTP API.

    Args:
        image_bytes (bytes): содержимое изображения

    Returns:
        str: label ('0'-'9', 'X' или 'unknown')
    """
    # Переводим изображение в Base64, т.к. API принимает JSON
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [image_b64]
    }

    try:
        # Отправляем запрос с потоковой обработкой
        response = requests.post(OLLAMA_API, json=payload, stream=True)
        response.raise_for_status()

        text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "response" in obj and obj["response"]:
                text += obj["response"]

        text = text.strip().upper()
        label = text[0] if text and text[0] in "0123456789X" else "unknown"
        return label

    except Exception as e:
        print("Ошибка при классификации изображения через API:", e)
        return "unknown"
