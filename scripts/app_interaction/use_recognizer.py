import json
import os
import sys
from PIL import Image

# Добавляем путь для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.recognizers.recognizer import DocumentRecognizer


def main():
    # Хардкодим путь к изображению
    image_path = "data/bio7.png"  # поменяй на свой путь

    if not os.path.exists(image_path):
        print(f"Ошибка: файл {image_path} не найден")
        return

    try:
        # Загружаем изображение
        image = Image.open(image_path)

        # Создаем распознаватель
        recognizer = DocumentRecognizer()

        # Распознаем
        result = recognizer.recognize(image)

        # Печатаем результат как есть
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"Ошибка при распознавании: {e}")


if __name__ == '__main__':
    main()