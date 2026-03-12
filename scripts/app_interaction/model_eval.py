import os
import json
import numpy as np
import cv2
from collections import defaultdict
from app.ml.loader import get_extended_model


# -------- НАСТРОЙКИ --------
DATASET_DIR = "data/dataset/dataset/val"   # папки 0..9 и x
OUTPUT_FILE = "cnn_threshold_data_val.json"


def load_28x28_binary(path):
    """
    Загружает уже подготовленное изображение 28x28.
    Никаких ресайзов и предобработки НЕ делаем.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    if img.shape != (28, 28):
        raise ValueError(f"{path} имеет размер {img.shape}, а не 28x28")

    # перевод в [0,1]
    img = img.astype(np.float32) / 255.0

    # (28,28) -> (28,28,1)
    img = np.expand_dims(img, axis=-1)

    # batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def main():
    print("Загрузка модели...")
    model = get_extended_model()
    print("Модель загружена")

    results = []

    total = 0
    correct = 0

    # статистика по классам
    per_class = defaultdict(lambda: {"total": 0, "correct": 0})

    for class_dir in sorted(os.listdir(DATASET_DIR)):
        class_path = os.path.join(DATASET_DIR, class_dir)
        if not os.path.isdir(class_path):
            continue

        if class_dir.isdigit():
            true_label = int(class_dir)
        elif class_dir.lower() == "x":
            true_label = 11
        else:
            continue

        print(f"\nКласс {class_dir}")

        for filename in os.listdir(class_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(class_path, filename)

            try:
                x = load_28x28_binary(path)
            except Exception as e:
                print("skip:", path, e)
                continue

            # --- ВАЖНО: это и есть чистый вывод CNN ---
            probs = model.predict(x, verbose=0)[0]

            pred_label = int(np.argmax(probs))
            confidence = float(np.max(probs))

            # margin (разница 1-го и 2-го классов)
            sorted_probs = np.sort(probs)
            second_best = float(sorted_probs[-2])
            margin = confidence - second_best

            is_correct = pred_label == true_label

            total += 1
            per_class[true_label]["total"] += 1

            if is_correct:
                correct += 1
                per_class[true_label]["correct"] += 1

            results.append({
                "file": filename,
                "true": true_label,
                "pred": pred_label,
                "confidence": confidence,
                "margin": margin,
                "correct": is_correct,
                "probs": probs.tolist()
            })

    accuracy = correct / total if total else 0

    print("\n======================")
    print("Всего:", total)
    print("Accuracy:", accuracy)

    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "total": total,
            "per_class": per_class,
            "results": results
        }, f, indent=2)

    print("Сохранено:", OUTPUT_FILE)


if __name__ == "__main__":
    main()