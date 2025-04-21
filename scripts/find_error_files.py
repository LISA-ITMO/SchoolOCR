import os
import shutil
import json

# Путь к папке с обработанными результатами
PROCESSED_RESULTS_DIR = "processed_lists_docker_rusnew"

# Папка для сохранения изображений с ошибками
ERROR_IMAGES_DIR = "processed_lists_docker_rusnew5"


def find_error_images():
    # Создаем папку для ошибок, если ее нет
    os.makedirs(ERROR_IMAGES_DIR, exist_ok=True)

    error_count = 0
    processed_folders = 0

    print("Начинаем поиск изображений с ошибками распознавания...")

    # Проходим по всем папкам в PROCESSED_RESULTS_DIR
    for folder_name in os.listdir(PROCESSED_RESULTS_DIR):
        folder_path = os.path.join(PROCESSED_RESULTS_DIR, folder_name)

        # Пропускаем папку с ошибками и не-папки
        if folder_name == "error_images" or not os.path.isdir(folder_path):
            continue

        processed_folders += 1

        # Ищем все подпапки с результатами
        for image_dir in os.listdir(folder_path):
            image_dir_path = os.path.join(folder_path, image_dir)
            json_path = os.path.join(image_dir_path, "response.json")

            # Проверяем наличие JSON-файла
            if not os.path.exists(json_path):
                continue

            # Читаем JSON и проверяем наличие ошибок
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if data.get("errors"):
                        # Нашли ошибку - ищем соответствующее изображение
                        for ext in ['.jpg', '.jpeg', '.png']:
                            img_path = os.path.join(image_dir_path, image_dir + ext)
                            if os.path.exists(img_path):
                                # Формируем уникальное имя для файла с ошибкой
                                dst_name = f"{folder_name}_{image_dir}{ext}"
                                dst_path = os.path.join(ERROR_IMAGES_DIR, dst_name)

                                # Копируем изображение
                                shutil.copy2(img_path, dst_path)
                                error_count += 1
                                print(f"Найдена ошибка в: {dst_name}")
                                break
                except json.JSONDecodeError as e:
                    print(f"Ошибка чтения JSON в {json_path}: {e}")
                    continue

    print("\n=== Результаты поиска ===")
    print(f"Обработано папок: {processed_folders}")
    print(f"Найдено изображений с ошибками: {error_count}")
    print(f"Все изображения с ошибками сохранены в: {ERROR_IMAGES_DIR}")


if __name__ == "__main__":
    find_error_images()