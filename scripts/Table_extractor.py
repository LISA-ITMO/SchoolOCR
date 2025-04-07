import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from wired_table_rec import WiredTableRecognition
from utils.mnist_preprocess_cell import preprocess_image

INPUT_DIR = "cropped_tables"
OUTPUT_DIR = "processed_tables"


def table_extraction(img_path, model, output_dir):
    wired_engine = WiredTableRecognition()
    table_engine = wired_engine
    img = cv2.imread(img_path)

    # Преобразуем изображение в градации серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Применение бинаризации
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Морфологические операции для соединения пунктирных линий
    kernel = np.ones((3, 3), np.uint8)  # Размер ядра можно настроить
    dilated = cv2.dilate(binary, kernel, iterations=2)  # Дилатация для соединения линий

    _, _, polygons, logic_points, _ = table_engine(dilated, need_ocr=False)

    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {dilated}.")
        return None

    second_row_cells = []
    for i, logic in enumerate(logic_points):
        if logic[0] == 1 and logic[1] == 1:
            second_row_cells.append({
                "index": i + 1,
                "coordinates": list(map(int, polygons[i])),
            })

    second_row_cells = second_row_cells[1:-2]

    page_name = os.path.splitext(os.path.basename(img_path))[0]
    page_output_dir = os.path.join(output_dir, page_name)
    os.makedirs(page_output_dir, exist_ok=True)

    results = []

    num_cells = len(second_row_cells)
    if num_cells == 0:
        print(f"Нет ячеек для обработки в файле {img_path}.")
        return None

    plt.figure(figsize=(15, 8))

    plt.subplot(3, 1, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Table")
    plt.axis('off')

    for i, cell_info in enumerate(second_row_cells):
        x1, y1, x2, y2 = cell_info["coordinates"]
        cell_img = img[y1:y2, x1:x2]

        if cell_img.size == 0:
            print(f"Ячейка {cell_info['index']} ({img_path}): пустая область, пропуск.")
            results.append({
                "index": cell_info["index"],
                "coordinates": cell_info["coordinates"],
                "content": None,
                "probability": 0.0,
            })
            continue

        input_data, processed_img = preprocess_image(cell_img)

        if input_data is None or processed_img is None:
            print(f"Ячейка {cell_info['index']} ({img_path}): не удалось обработать изображение.")
            results.append({
                "index": cell_info["index"],
                "coordinates": cell_info["coordinates"],
                "content": None,
                "probability": 0.0,
            })
            continue

        predictions = model.predict(input_data)
        predicted_digit = np.argmax(predictions)
        predicted_prob = float(np.max(predictions))

        results.append({
            "index": cell_info["index"],
            "coordinates": cell_info["coordinates"],
            "content": int(predicted_digit),
            "probability": predicted_prob,
        })

        print(
            f"Ячейка {cell_info['index']} ({img_path}): распознана цифра {predicted_digit} с вероятностью {predicted_prob:.4f}")

        plt.subplot(3, num_cells, num_cells + i + 1)
        plt.imshow(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original {cell_info['index']}")
        plt.axis('off')

        plt.subplot(3, num_cells, 2 * num_cells + i + 1)
        plt.imshow(processed_img, cmap='gray')
        plt.title(f"Processed {cell_info['index']}\nPred: {predicted_digit}\nProb: {predicted_prob:.4f}")
        plt.axis('off')

    page_plot_file = f"{page_output_dir}/table.png"
    plt.tight_layout()
    plt.savefig(page_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Обработанные данные для {img_path} сохранены в файл: {page_plot_file}")

    output_json = {
        "image_path": img_path,
        "total_cells": len(second_row_cells),
        "cells": results,
        "plot_image_path": os.path.relpath(page_plot_file, output_dir),
    }

    return output_json


if __name__ == "__main__":
    model = tf.keras.models.load_model("../mnist_recognation_extendend.h5")
    print("Модель успешно загружена.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(INPUT_DIR, filename)
            print(f"Обработка файла: {img_path}")

            try:
                output_json = table_extraction(img_path, model, OUTPUT_DIR)

                if output_json is not None:
                    json_file = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0],
                                             f"{os.path.splitext(filename)[0]}.json")
                    os.makedirs(os.path.dirname(json_file), exist_ok=True)
                    with open(json_file, "w") as f:
                        json.dump(output_json, f, indent=4)
                    print(f"Результаты для {filename} сохранены в файл: {json_file}")

            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {e}")
