import json
import os
import sys
import cv2
import numpy as np
from PIL import Image

# Добавляем путь для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.table_recognizer import TableRecognizer


def main():
    # Хардкодим путь к изображению
    image_path = "data/РУС 8 кл 1 в 39_page_1.jpg"  # поменяй на свой путь

    if not os.path.exists(image_path):
        print(f"Ошибка: файл {image_path} не найден")
        return

    try:
        # Загружаем изображение
        pil_image = Image.open(image_path)

        # Конвертируем PIL в numpy array для OpenCV
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Создаем распознаватель с включенным debug
        recognizer = TableRecognizer(debug=True, save_cells=True, save_dir="./output_cells")

        # Распознаем
        result = recognizer.recognize_scores_table(image)

        # Печатаем результат как есть
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"Ошибка при распознавании: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
