import os
import requests
import base64
import json

# Адрес сервера
SERVER_URL = "http://localhost:8000/recognize"

# Пути к папкам
INPUT_IMAGES_DIR = "output_images"  # Папка с исходными изображениями
PROCESSED_LISTS_DIR = "processed_lists"  # Папка для сохранения результатов

# Функция для кодирования изображения в base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Функция для отправки запроса на сервер
def send_image_to_server(image_base64):
    # Формируем JSON-тело запроса
    payload = {"image_base64": image_base64}

    try:
        # Отправляем POST-запрос на сервер
        response = requests.post(SERVER_URL, json=payload)

        # Проверяем статус ответа
        if response.status_code == 200:
            return response.json()  # Возвращаем JSON-ответ
        else:
            print(f"Ошибка: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Произошла ошибка при отправке запроса: {e}")
        return None

# Функция для сохранения результата
def save_result(image_path, server_response):
    # Извлекаем имя файла без расширения
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Создаем папку с именем файла
    result_dir = os.path.join(PROCESSED_LISTS_DIR, image_name)
    os.makedirs(result_dir, exist_ok=True)

    # Сохраняем исходное изображение
    saved_image_path = os.path.join(result_dir, os.path.basename(image_path))
    with open(saved_image_path, "wb") as f:
        with open(image_path, "rb") as img_file:
            f.write(img_file.read())
    print(f"Изображение сохранено: {saved_image_path}")

    # Сохраняем JSON-ответ сервера
    json_path = os.path.join(result_dir, "response.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(server_response, json_file, indent=4, ensure_ascii=False)
    print(f"JSON-ответ сохранен: {json_path}")

# Основная функция
def main():
    # Проверяем, существует ли папка с изображениями
    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"Папка {INPUT_IMAGES_DIR} не найдена.")
        return

    # Получаем список всех изображений в папке
    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"В папке {INPUT_IMAGES_DIR} нет изображений.")
        return

    # Обрабатываем каждое изображение
    for image_file in image_files:
        image_path = os.path.join(INPUT_IMAGES_DIR, image_file)
        print(f"Обработка изображения: {image_path}")

        # Кодируем изображение в base64
        image_base64 = encode_image_to_base64(image_path)
        print("Изображение успешно закодировано в base64.")

        # Отправляем изображение на сервер
        print("Отправка изображения на сервер...")
        server_response = send_image_to_server(image_base64)

        if server_response:
            print("Ответ от сервера получен:")
            print(json.dumps(server_response, indent=4, ensure_ascii=False))

            # Сохраняем результат
            save_result(image_path, server_response)
        else:
            print(f"Не удалось получить ответ от сервера для изображения: {image_path}")

if __name__ == "__main__":
    main()