import os
import json

# Путь к папке с JSON-файлами
JSON_DIR = "processed_lists_history"


# Функция для подсчета ошибок и предупреждений
def count_errors_and_warnings_in_jsons(json_dir):
    error_count = 0  # Счетчик ошибок
    warning_count = 0  # Счетчик предупреждений

    # Проверяем, существует ли папка
    if not os.path.exists(json_dir):
        print(f"Папка {json_dir} не найдена.")
        return

    # Получаем список всех JSON-файлов в папке
    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)

                try:
                    # Читаем JSON-файл
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Проверяем поле "errors"
                    if data.get("errors") is not None:  # Если "errors" не равно null
                        error_count += 1
                        print(f"Файл с ошибками: {json_path}")

                    # Проверяем поле "warnings"
                    if data.get("warnings") is not None:  # Если "warnings" не равно null
                        warning_count += 1
                        print(f"Файл с предупреждениями: {json_path}")

                except Exception as e:
                    print(f"Ошибка при обработке файла {json_path}: {e}")

    # Выводим итоговый результат
    print(f"Количество JSON-файлов с ошибками (errors != null): {error_count}")
    print(f"Количество JSON-файлов с предупреждениями (warnings != null): {warning_count}")


# Основная функция
if __name__ == "__main__":
    count_errors_and_warnings_in_jsons(JSON_DIR)