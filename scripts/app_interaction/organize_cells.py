import os
import sys
import cv2
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.cell_recognizer import CellRecognizer


def organize_cells_by_digit(input_dir, output_dir):
    for i in range(12):
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

    os.makedirs(os.path.join(output_dir, "unknown"), exist_ok=True)

    print("Инициализация CellRecognizer...")
    recognizer = CellRecognizer(debug=False, enable_logging=False)
    print("CellRecognizer инициализирован")

    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".png"):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"PNG файлы не найдены в папке {input_dir}")
        return

    print(f"Найдено PNG файлов: {len(image_files)}")

    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print(f"[{i}/{len(image_files)}] Обрабатываем {filename}")

        try:
            image = cv2.imread(image_path)
            if image is None:
                print("  Не удалось загрузить изображение")
                continue

            res = recognizer.recognize_cell(image, filename)
            digit = res.digit
            prob = float(res.prob or 0.0)

            if digit is None:
                target_dir = os.path.join(output_dir, "unknown")
                name_without_ext = os.path.splitext(filename)[0]
                new_filename = f"{name_without_ext}_prob{prob:.3f}.png"
                target_path = os.path.join(target_dir, new_filename)
                shutil.copy2(image_path, target_path)
                print("  Не удалось распознать цифру -> unknown")
                continue

            target_dir = os.path.join(output_dir, str(digit))
            name_without_ext = os.path.splitext(filename)[0]
            new_filename = f"{name_without_ext}_prob{prob:.3f}.png"
            target_path = os.path.join(target_dir, new_filename)
            shutil.copy2(image_path, target_path)
            print(f"  Распознано: {digit} (вероятность: {prob:.3f}) -> {target_dir}")

        except Exception as e:
            print(f"  Ошибка при обработке {image_path}: {e}")

    print("\n" + "=" * 50)
    print("Организация завершена!")
    print(f"Входная папка: {input_dir}")
    print(f"Выходная папка: {output_dir}")
    print(f"Обработано файлов: {len(image_files)}")
    print("=" * 50)


def main():
    input_dir = "extracted_cells"
    output_dir = "organized_cells_new"

    if not os.path.exists(input_dir):
        print(f"Ошибка: входная папка {input_dir} не существует")
        return

    os.makedirs(output_dir, exist_ok=True)
    organize_cells_by_digit(input_dir, output_dir)


if __name__ == "__main__":
    main()
