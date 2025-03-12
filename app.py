from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import json
import tensorflow as tf
from wired_table_rec import WiredTableRecognition
from mnist_preprocess_image import preprocess_image
import pytesseract

app = Flask(__name__)

# Загрузка конфига и модели
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

config = load_config()
model = tf.keras.models.load_model("mnist_model.keras")

# Функция для извлечения региона
def extract_region(image, coords):
    x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
    region_img = image[y1:y2, x1:x2]
    return region_img

# Функция для распознавания текста в шапке
def recognize_hat(region_img):
    text = pytesseract.image_to_string(region_img, lang="rus").strip()
    print(f"Recognized hat text: {text}")
    return text

# Функция для распознавания кода
def recognize_code(region_img):
    code = pytesseract.image_to_string(region_img, config="--psm 6").strip()
    print(f"Recognized code: {code}")
    return code if code.isdigit() else None

# Функция для распознавания таблицы
def recognize_table(region_img, model):
    wired_engine = WiredTableRecognition()
    _, _, polygons, logic_points, _ = wired_engine(region_img, need_ocr=False)
    second_row_cells = []
    for i, logic in enumerate(logic_points):
        if logic[0] == 1 and logic[1] == 1:  # Фильтр для второй строки
            second_row_cells.append({
                "index": i + 1,
                "coordinates": list(map(int, polygons[i])),
            })

    # Фильтрация ячеек (пример для 13 ячеек)
    second_row_cells = second_row_cells[1:-2]

    results = []
    for cell_info in second_row_cells:
        x1, y1, x2, y2 = cell_info["coordinates"]
        cell_img = region_img[y1:y2, x1:x2]

        input_data, _ = preprocess_image(cell_img)
        if input_data is None:
            results.append({
                "index": cell_info["index"],
                "coordinates": cell_info["coordinates"],
                "content": None,
                "probability": 0.0
            })
            continue

        predictions = model.predict(input_data)
        predicted_digit = int(np.argmax(predictions))
        predicted_prob = float(np.max(predictions))

        results.append({
            "index": cell_info["index"],
            "coordinates": cell_info["coordinates"],
            "content": predicted_digit,
            "probability": predicted_prob
        })

    return {
        "total_cells": len(results),
        "cells": results
    }

# Основная функция обработки изображения
def process_image(image, config, model):
    doc_type = "math_6_grade"
    regions_config = config[doc_type]["regions"]

    hat_region = extract_region(image, config["regions"]["hat"])
    code_region = extract_region(image, config["regions"]["code"])
    table_region = extract_region(image, regions_config["table"])

    table_result = recognize_table(table_region, model)

    return {
        "hat": recognize_hat(hat_region),
        "code": recognize_code(code_region),
        "table": table_result
    }

# API endpoint для обработки изображения
@app.route('/process-image', methods=['POST'])
def process_image_api():
    try:
        # Получаем base64 из запроса
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Декодируем base64 в изображение
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Обрабатываем изображение
        result = process_image(image, config, model)

        # Возвращаем результат
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Запуск Flask приложения
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)