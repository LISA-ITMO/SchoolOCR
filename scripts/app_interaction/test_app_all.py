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
INPUT_ROOT_DIR = "../check_school/Okasana"  # Корневая папка с подпапками для обработки
PROCESSED_LISTS_DIR = (
    "../check_school/Okasana_proccessed"  # Папка для сохранения результатов
)


class RequestStats:
    """
    Stores and updates statistics about requests made to different folders."""

    def __init__(self):
        """
        Increments the success counters for a request.

            Args:
                folder_name: The name of the folder associated with the successful request.

            Returns:
                None
        """
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.error_codes = {}
        self.recognition_errors = 0
        self.start_time = datetime.now()
        self.folder_stats = {}  # Статистика по каждой папке

    def add_success(self, folder_name):
        """
        Adds a failure to the specified folder's statistics.

            Args:
                folder_name: The name of the folder where the failure occurred.
                status_code: The HTTP status code associated with the failure (optional).

            Returns:
                None
        """
        self.total_requests += 1
        self.successful_requests += 1
        self._update_folder_stats(folder_name, success=True)

    def add_failure(self, folder_name, status_code=None):
        """
        Increments the failure count and updates related statistics.

            Args:
                folder_name: The name of the folder where the failure occurred.
                status_code: An optional HTTP status code associated with the failure.

            Returns:
                None
        """
        self.total_requests += 1
        self.failed_requests += 1
        if status_code:
            self.error_codes[status_code] = self.error_codes.get(status_code, 0) + 1
        self._update_folder_stats(folder_name, success=False)

    def add_recognition_error(self, folder_name):
        """
        Updates the statistics for a given folder.

            Args:
                folder_name: The name of the folder to update stats for.
                success: A boolean indicating if processing was successful (optional).
                recognition_error: A flag indicating whether a recognition error occurred (optional).

            Returns:
                None
        """
        self.recognition_errors += 1
        self._update_folder_stats(folder_name, recognition_error=True)

    def _update_folder_stats(self, folder_name, success=None, recognition_error=False):
        """
        Updates the statistics for a given folder.

            Args:
                folder_name: The name of the folder to update stats for.
                success: A boolean indicating if the operation was successful.
                recognition_error: A boolean indicating if a recognition error occurred.

            Returns:
                None
        """
        if folder_name not in self.folder_stats:
            self.folder_stats[folder_name] = {
                "total": 0,
                "success": 0,
                "failed": 0,
                "recognition_errors": 0,
            }

        self.folder_stats[folder_name]["total"] += 1
        if success is True:
            self.folder_stats[folder_name]["success"] += 1
        elif success is False:
            self.folder_stats[folder_name]["failed"] += 1

        if recognition_error:
            self.folder_stats[folder_name]["recognition_errors"] += 1

    def print_stats(self):
        """
        Prints statistics about the requests made.

            Args:
                duration: The total time taken for all requests.

            Returns:
                None
        """
        duration = datetime.now() - self.start_time
        print("\n=== Общая статистика обработки ===")
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

        # Вывод статистики по папкам
        print("\n=== Статистика по папкам ===")
        for folder_name, stats in self.folder_stats.items():
            print(f"\nПапка: {folder_name}")
            print(f"  Всего изображений: {stats['total']}")
            print(
                f"  Успешно обработано: {stats['success']} ({stats['success'] / stats['total']:.1%})"
            )
            print(
                f"  Ошибок при обработке: {stats['failed']} ({stats['failed'] / stats['total']:.1%})"
            )
            if stats["recognition_errors"]:
                print(f"  Ошибок распознавания: {stats['recognition_errors']}")


stats = RequestStats()


def encode_image_to_base64(image_path):
    """
    Encodes an image to a base64 string.

        Args:
            image_path: The path to the image file.

        Returns:
            str: The base64 encoded string of the image, or None if encoding fails.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def send_image_to_server(image_base64):
    """
    Sends an image to a specified server.

        Args:
            image_path: The path to the image file.
            server_url: The URL of the server to send the image to.

        Returns:
            None: Returns None if the request is successful or encounters an error.  Prints any response text or error messages to the console.
    """
    payload = {"image_base64": image_base64}
    headers = {"Authorization": API_KEY}

    try:
        response = requests.post(SERVER_URL, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Ошибка: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Произошла ошибка при отправке запроса: {e}")
        return None


def save_result(folder_name, image_path, server_response):
    """
    Saves the server response and the processed image to a specified directory.

        Args:
            server_response: The JSON response from the server.
            image_path: The path to the original image file.
            er_result_dir: The base directory for saving results.
            image_name:  The name of the image (used in constructing the result directory).

        Returns:
            None
    """
    # Создаем подпапку с именем исходной папки в PROCESSED_LISTS_DIR
    folder_result_dir = os.path.join(PROCESSED_LISTS_DIR, folder_name)
    os.makedirs(folder_result_dir, exist_ok=True)

    # Создаем подпапку для конкретного изображения
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_dir = os.path.join(folder_result_dir, image_name)
    os.makedirs(result_dir, exist_ok=True)

    # Сохраняем изображение
    saved_image_path = os.path.join(result_dir, os.path.basename(image_path))
    with open(saved_image_path, "wb") as f:
        with open(image_path, "rb") as img_file:
            f.write(img_file.read())

    # Сохраняем JSON-ответ
    json_path = os.path.join(result_dir, "response.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(server_response, json_file, indent=4, ensure_ascii=False)


def process_folder(folder_path, folder_name):
    """
    Processes all images in a given folder, sends them to a server for recognition, and saves the results.

        Args:
            folder_name: The name of the folder to process.

        Returns:
            None
    """
    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))
    ]
    if not image_files:
        print(f"В папке {folder_name} нет изображений.")
        return

    print(f"\nОбработка папки: {folder_name} ({len(image_files)} изображений)")

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, image_file)
        print(f"[{idx}/{len(image_files)}] Обработка изображения: {image_file}")

        image_base64 = encode_image_to_base64(image_path)
        server_response = send_image_to_server(image_base64)

        if server_response:
            # Проверка наличия ошибок в ответе сервера
            if "errors" in server_response and server_response["errors"]:
                stats.add_recognition_error(folder_name)
                print(f"Возникли ошибки распознавания: {server_response['errors']}")
            else:
                print("Ошибок распознавания не обнаружено.")

            stats.add_success(folder_name)
            save_result(folder_name, image_path, server_response)
            print("Результат успешно сохранен")
        else:
            stats.add_failure(folder_name)
            print("Не удалось обработать изображение")


def main():
    """
    Main function to process a folder and print statistics.

        This function calls `process_folder` with the specified folder path and name,
        then prints the collected statistics using the `print_stats` method of the stats object.

        Args:
            folder_path: The path to the folder to be processed.
            folder_name: The name of the folder being processed.

        Returns:
            None
    """
    if not os.path.exists(INPUT_ROOT_DIR):
        print(f"Корневая папка {INPUT_ROOT_DIR} не найдена.")
        return

    # Создаем папку для результатов, если ее нет
    os.makedirs(PROCESSED_LISTS_DIR, exist_ok=True)

    # Получаем список всех подпапок в INPUT_ROOT_DIR
    folders_to_process = [
        f
        for f in os.listdir(INPUT_ROOT_DIR)
        if os.path.isdir(os.path.join(INPUT_ROOT_DIR, f))
    ]

    if not folders_to_process:
        print(f"В папке {INPUT_ROOT_DIR} нет подпапок для обработки.")
        return

    print(f"Найдено {len(folders_to_process)} папок для обработки")

    for folder_name in folders_to_process:
        folder_path = os.path.join(INPUT_ROOT_DIR, folder_name)
        process_folder(folder_path, folder_name)

    stats.print_stats()


if __name__ == "__main__":
    main()
