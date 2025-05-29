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
INPUT_IMAGES_DIR = "../to_proccess/help_imgs/litr_new"
PROCESSED_LISTS_DIR = "../proccessed_/processed_lists_docker_litrnew"


class RequestStats:
    """
    Tracks statistics related to requests made to an external service."""

    def __init__(self):
        """
        Increments the count of successful requests.

            Args:
                None

            Returns:
                None
        """
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.error_codes = {}
        self.recognition_errors = 0
        self.start_time = datetime.now()

    def add_success(self):
        """
        Increments the counters for total and successful requests.

            This method is called to record a successful request made to an external service.
            It updates both the total number of requests and the number of successfully completed requests.

            Args:
                None

            Returns:
                None
        """
        self.total_requests += 1
        self.successful_requests += 1

    def add_failure(self, status_code=None):
        """
        Increments the counters for total and failed requests.

            Updates internal statistics to reflect a failed request,
            optionally tracking the HTTP status code if provided.

            Args:
                s_code: The HTTP status code of the failed request (optional).

            Returns:
                None
        """
        self.total_requests += 1
        self.failed_requests += 1
        if status_code:
            self.error_codes[status_code] = self.error_codes.get(status_code, 0) + 1

    def add_recognition_error(self):
        """
        Increments the count of recognition errors.

            Args:
                None

            Returns:
                None
        """
        self.recognition_errors += 1

    def print_stats(self):
        """
        Prints statistics about the request processing.

            Args:
                None

            Returns:
                None
        """
        duration = datetime.now() - self.start_time
        print("\n=== Статистика обработки ===")
        print(f"Всего запросов: {self.total_requests}")
        print(
            f"Успешных: {self.successful_requests} ({self.successful_requests / self.total_requests:.1%})"
        )
        print(
            f"Неудачных: {self.failed_requests} ({self.failed_requests / self.total_requests:.1%})"
        )
        if self.error_codes:
            print("Коды ошибок:")
            for code, count in self.error_codes.items():
                print(f"  {code}: {count} раз")
        if self.recognition_errors:
            print(f"Ошибки распознавания (errors != null): {self.recognition_errors}")
        print(f"Общее время выполнения: {duration.total_seconds():.2f} сек")
        print(
            f"Среднее время на запрос: {duration.total_seconds() / self.total_requests:.2f} сек"
        )


stats = RequestStats()


def encode_image_to_base64(image_path):
    """
    Sends an image encoded as a base64 string to the server.

        Args:
            image_base64: The base64 encoded image string.

        Returns:
            The response from the server (details not specified in provided code).
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def send_image_to_server(image_base64):
    """
    Sends image data to the server and returns the response.

        Args:
            payload: The JSON payload containing image data.
            headers: The HTTP headers for the request.
            stats: An object used to track success/failure statistics.

        Returns:
            dict: The JSON response from the server if successful, otherwise None.
    """
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
    """
    Saves the image and server response to a specified directory.

        Args:
            image_path: The path to the original image file.
            server_response: The server's response data (typically a dictionary).
            result_dir: The directory where the image and JSON response will be saved.

        Returns:
            None
    """
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
    """
    Processes images from a directory, sends them to a server for recognition, and saves the results.

        This function iterates through all image files (png, jpg, jpeg, pdf) in the INPUT_IMAGES_DIR directory,
        encodes each image to base64, sends it to a server for processing, handles potential errors in the server response,
        and saves the result if successful. It also tracks and prints statistics about the process.

        Parameters:
            None

        Returns:
            None
    """
    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"Папка {INPUT_IMAGES_DIR} не найдена.")
        return

    image_files = [
        f
        for f in os.listdir(INPUT_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))
    ]
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
