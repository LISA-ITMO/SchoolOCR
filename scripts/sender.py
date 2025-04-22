import requests
import base64
import json

# Адрес сервера
SERVER_URL = "http://158.160.57.65:8000/recognize"
# SERVER_URL = "http://localhost:8000/recognize"

# Путь к изображению
IMAGE_PATH = "help_imgs/rus_new5/page_8.jpg"

# API-ключ
API_KEY = ""

# Функция для кодирования изображения в base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Функция для отправки запроса на сервер
def send_image_to_server(image_base64):
    # Формируем JSON-тело запроса
    payload = {"image_base64": image_base64}

    # Заголовки с API-ключом
    headers = {"Authorization": API_KEY}

    # Отправляем POST-запрос на сервер
    response = requests.post(SERVER_URL, json=payload, headers=headers)

    # Проверяем статус ответа
    if response.status_code == 200:
        return response.json()  # Возвращаем JSON-ответ
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)
        return None

# Основная функция
def main():
    # Кодируем изображение в base64
    image_base64 = encode_image_to_base64(IMAGE_PATH)
    print("Изображение успешно закодировано в base64.")

    # Отправляем изображение на сервер
    print("Отправка изображения на сервер...")
    result = send_image_to_server(image_base64)

    # Выводим результат
    if result:
        print("Ответ от сервера:")
        print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
