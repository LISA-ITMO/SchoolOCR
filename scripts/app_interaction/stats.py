import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict


def analyze_json_files(folder_path):
    # Собираем статистику по всем файлам
    stats = defaultdict(int)
    total_files = 0

    # Рекурсивно ищем все JSON-файлы в папке и подпапках
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # Проверяем, что файл соответствует ожидаемой структуре
                        if 'scores' in data and isinstance(data['scores'], dict):
                            total_files += 1
                            for task, score_data in data['scores'].items():
                                if isinstance(score_data, list) and len(score_data) >= 2:
                                    score = score_data[1]  # Второй элемент - точность распознавания
                                    if score >= 0.9:
                                        stats['perfect'] += 1
                                    elif score >= 0.8:
                                        stats['good'] += 1
                                    elif score >= 0.7:
                                        stats['acceptable'] += 1
                                    else:
                                        stats['poor'] += 1
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Ошибка при чтении файла {file_path}: {e}")

    return stats, total_files


def plot_statistics(stats, total_files):
    if total_files == 0:
        print("Не найдено ни одного подходящего JSON-файла.")
        return

    # Подготовка данных для гистограммы
    categories = ['Perfect (0.9-1.0)', 'Good (0.8-0.89)', 'Acceptable (0.7-0.79)', 'Poor (<0.7)']
    counts = [
        stats.get('perfect', 0),
        stats.get('good', 0),
        stats.get('acceptable', 0),
        stats.get('poor', 0)
    ]

    # Создание гистограммы
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, counts, color=['green', 'lightgreen', 'orange', 'red'])

    # Добавление подписей
    plt.title('Статистика распознавания цифр', fontsize=14)
    plt.xlabel('Категории качества распознавания', fontsize=12)
    plt.ylabel('Количество цифр', fontsize=12)

    # Добавление значений на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height} ({height / sum(counts) * 100:.1f}%)',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder_path = "../proccessed_/processed_lists_docker"

    if not os.path.isdir(folder_path):
        print("Указанная папка не существует.")
    else:
        stats, total_files = analyze_json_files(folder_path)
        print(f"\nПроанализировано файлов: {total_files}")
        print(f"Всего таблиц: {sum(stats.values())}")
        print("\nСтатистика распознавания:")
        print(f"Perfect (0.9-1.0): {stats.get('perfect', 0)}")
        print(f"Good (0.8-0.89): {stats.get('good', 0)}")
        print(f"Acceptable (0.7-0.89): {stats.get('acceptable', 0)}")
        print(f"Poor (<0.7): {stats.get('poor', 0)}")

        plot_statistics(stats, total_files)