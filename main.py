import json
import cv2
import pytesseract
import numpy as np
import tensorflow as tf
from wired_table_rec import WiredTableRecognition
from mnist_preprocess_image import preprocess_image


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def extract_region(image, coords):
    x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
    region_img = image[y1:y2, x1:x2]
    return region_img


def recognize_hat(region_img):
    text = pytesseract.image_to_string(region_img, lang="rus").strip()
    print(f"Recognized hat text: {text}")
    return text


def recognize_code(region_img):
    code = pytesseract.image_to_string(region_img, config="--psm 6").strip()
    print(f"Recognized code: {code}")
    return code if code.isdigit() else None


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


def process_image(image_path, config, model):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Ошибка загрузки изображения: {image_path}")

    doc_type = "math_6_grade"
    regions_config = config[doc_type]["regions"]

    hat_region = extract_region(image, config["regions"]["hat"])
    code_region = extract_region(image, config["regions"]["code"])
    table_region = extract_region(image, regions_config["table"])

    table_result = recognize_table(
        table_region,
        model,
    )

    return {
        "hat": recognize_hat(hat_region),
        "code": recognize_code(code_region),
        "table": table_result
    }


if __name__ == "__main__":
    config = load_config()
    model = tf.keras.models.load_model("mnist_model.keras")

    output = process_image("./output_images/page_2.jpg", config, model)

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)