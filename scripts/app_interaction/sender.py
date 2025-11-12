import requests
import json

# Адрес сервера
SERVER_URL = "http://localhost:8000/recognize"

# Путь к изображению
IMAGE_PATH = "data/РУС 8 кл 1 в 39_page_1.jpg"

# API-ключ
API_KEY = "42d354f4b6e38ff95553137e49f724c9bc429399"


# Функция для отправки файла на сервер
def send_file_to_server(file_path):
    # Открываем файл в бинарном режиме
    with open(file_path, "rb") as file:
        # Создаем словарь с файлом для отправки
        files = {"file": (file_path, file, "image/jpeg")}

        # Заголовки с API-ключом
        headers = {"Authorization": API_KEY}

        # Отправляем POST-запрос на сервер с файлом
        response = requests.post(SERVER_URL, files=files, headers=headers)

    # Проверяем статус ответа
    if response.status_code == 200:
        return response.json()  # Возвращаем JSON-ответ
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)
        return None


# Основная функция
def main():
    # Отправляем файл на сервер
    print("Отправка файла на сервер...")
    result = send_file_to_server(IMAGE_PATH)

    # Выводим результат
    if result:
        print("Ответ от сервера:")
        print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()