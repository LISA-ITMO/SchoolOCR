import os
import sys
import cv2
import numpy as np
import shutil
from PIL import Image

# Добавляем путь для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.cell_recognizer import CellRecognizer


def organize_cells_by_digit(input_dir, output_dir):
    """
    Организует ячейки по папкам с цифрами на основе распознавания
    """
    # Создаем папки для всех цифр (0-11)
    for i in range(12):
        digit_dir = os.path.join(output_dir, str(i))
        os.makedirs(digit_dir, exist_ok=True)

    # Создаем распознаватель
    print("Инициализация CellRecognizer...")
    recognizer = CellRecognizer(debug=False)
    print("CellRecognizer инициализирован")

    # Находим все PNG файлы
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                image_files.append(full_path)

    if not image_files:
        print(f"PNG файлы не найдены в папке {input_dir}")
        return

    print(f"Найдено PNG файлов: {len(image_files)}")

    # Обрабатываем каждое изображение
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Обрабатываем {os.path.basename(image_path)}")

        try:
            # Загружаем изображение
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Не удалось загрузить изображение")
                continue

            # Распознаем цифру
            filename = os.path.basename(image_path)
            digit, prob, prob_all = recognizer.recognize_cell(image, filename)

            if digit is not None:
                # Определяем папку назначения
                target_dir = os.path.join(output_dir, str(digit))

                # Создаем новое имя файла с вероятностью
                name_without_ext = os.path.splitext(filename)[0]
                new_filename = f"{name_without_ext}_prob{prob:.3f}.png"
                target_path = os.path.join(target_dir, new_filename)

                # Копируем файл
                shutil.copy2(image_path, target_path)
                print(f"  Распознано: {digit} (вероятность: {prob:.3f}) -> {target_dir}")
            else:
                print(f"  Не удалось распознать цифру")

        except Exception as e:
            print(f"  Ошибка при обработке {image_path}: {e}")

    print(f"\n{'=' * 50}")
    print("Организация завершена!")
    print(f"Входная папка: {input_dir}")
    print(f"Выходная папка: {output_dir}")
    print(f"Обработано файлов: {len(image_files)}")
    print(f"{'=' * 50}")


def main():
    # ХАРДКОД ПУТЕЙ
    input_dir = "extracted_cells"  # Папка с извлеченными ячейками
    output_dir = "organized_cells"  # Папка для организованных ячеек

    if not os.path.exists(input_dir):
        print(f"Ошибка: входная папка {input_dir} не существует")
        return

    # Создаем папку для результатов
    os.makedirs(output_dir, exist_ok=True)

    # Организуем ячейки по цифрам
    organize_cells_by_digit(input_dir, output_dir)


if __name__ == '__main__':
    main()