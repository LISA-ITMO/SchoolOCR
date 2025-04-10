import os
import requests
import base64
import json
from datetime import datetime

# Адрес сервера
SERVER_URL = "http://localhost:8000/recognize"

# API-ключ
API_KEY = ""

# Пути к папкам
INPUT_IMAGES_DIR = "scans_jpg/БИО 7 кл 1в 40 стр"
PROCESSED_LISTS_DIR = "processed_lists_docker_bio7"


class RequestStats:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.error_codes = {}
        self.recognition_errors = 0
        self.start_time = datetime.now()

    def add_success(self):
        self.total_requests += 1
        self.successful_requests += 1

    def add_failure(self, status_code=None):
        self.total_requests += 1
        self.failed_requests += 1
        if status_code:
            self.error_codes[status_code] = self.error_codes.get(status_code, 0) + 1

    def add_recognition_error(self):
        self.recognition_errors += 1

    def print_stats(self):
        duration = datetime.now() - self.start_time
        print("\n=== Статистика обработки ===")
        print(f"Всего запросов: {self.total_requests}")
        print(f"Успешных: {self.successful_requests} ({self.successful_requests / self.total_requests:.1%})")
        print(f"Неудачных: {self.failed_requests} ({self.failed_requests / self.total_requests:.1%})")
        if self.error_codes:
            print("Коды ошибок:")
            for code, count in self.error_codes.items():
                print(f"  {code}: {count} раз")
        if self.recognition_errors:
            print(f"Ошибки распознавания (errors != null): {self.recognition_errors}")
        print(f"Общее время выполнения: {duration.total_seconds():.2f} сек")
        print(f"Среднее время на запрос: {duration.total_seconds() / self.total_requests:.2f} сек")


stats = RequestStats()


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def send_image_to_server(image_base64):
    payload = {"image_base64": image_base64}

    # Заголовки с API-ключом
    headers = {"Authorization": API_KEY}

    try:
        response = requests.post(SERVER_URL, json=payload, headers=headers, timeout=10)

        if response.status_code == 200:
            stats.add_success()
            return response.json()
        else:
            stats.add_failure(response.status_code)
            print(f"Ошибка: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        stats.add_failure()
        print(f"Произошла ошибка при отправке запроса: {e}")
        return None


def save_result(image_path, server_response):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_dir = os.path.join(PROCESSED_LISTS_DIR, image_name)
    os.makedirs(result_dir, exist_ok=True)

    saved_image_path = os.path.join(result_dir, os.path.basename(image_path))
    with open(saved_image_path, "wb") as f:
        with open(image_path, "rb") as img_file:
            f.write(img_file.read())

    json_path = os.path.join(result_dir, "response.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(server_response, json_file, indent=4, ensure_ascii=False)


def main():
    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"Папка {INPUT_IMAGES_DIR} не найдена.")
        return

    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"В папке {INPUT_IMAGES_DIR} нет изображений.")
        return

    print(f"Найдено {len(image_files)} изображений для обработки")

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(INPUT_IMAGES_DIR, image_file)
        print(f"\n[{idx}/{len(image_files)}] Обработка изображения: {image_path}")

        image_base64 = encode_image_to_base64(image_path)
        server_response = send_image_to_server(image_base64)

        if server_response:
            # Проверка наличия ошибок в ответе сервера
            if "errors" in server_response and server_response["errors"]:
                stats.add_recognition_error()
                print(f"Возникли ошибки распознавания: {server_response['errors']}")
            else:
                print("Ошибок распознавания не обнаружено.")

            save_result(image_path, server_response)
            print("Результат успешно сохранен")
        else:
            print("Не удалось обработать изображение")

    stats.print_stats()


if __name__ == "__main__":
    main()