import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Путь к папке с JSON-файлами
JSON_DIR = "../check_school/Nailya_proccessed"
OUTPUT_DIR = os.path.join(JSON_DIR, "analytics")

# Создаем структуру папок для сохранения графиков
os.makedirs(os.path.join(OUTPUT_DIR, "confidence_histograms"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "recognition_status"), exist_ok=True)


def analyze_folder(folder_path, folder_name=""):
    confidences = []
    recognized_tables = 0
    unrecognizable_tables = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)

                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Участник: код участника
                    if data.get("participant_code") is not None:
                        recognized_tables += 1
                    else:
                        unrecognizable_tables += 1

                    # Оценки: confidence
                    scores = data.get("scores", {})
                    for key, value in scores.items():
                        if isinstance(value, list) and len(value) >= 2 and value[1] is not None:
                            confidences.append(value[1])

                except Exception as e:
                    print(f"Ошибка при обработке файла {json_path}: {e}")

    return confidences, recognized_tables, unrecognizable_tables


def plot_confidence_histogram(confidences, output_path, title_suffix=""):
    if not confidences:
        return

    bins = np.linspace(0, 1, 11)
    bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    counts, edges = np.histogram(confidences, bins=bins)

    # Переворачиваем данные для отображения от 1 до 0
    bin_labels = bin_labels[::-1]
    counts = counts[::-1]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_labels, counts, color='skyblue', edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 str(int(height)), ha='center', va='bottom')

    title = 'Распределение уверенности в распознавании оценок'
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title)
    plt.xlabel('Интервалы уверенности')
    plt.ylabel('Количество значений')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_recognition_status(recognized, unrecognized, output_path, title_suffix=""):
    labels = ['Распознанные', 'Нераспознанные']
    values = [recognized, unrecognized]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=['green', 'red'], edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 str(int(height)), ha='center', va='bottom')

    title = 'Количество распознанных и нераспознанных таблиц'
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title)
    plt.ylabel('Число таблиц')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_and_plot(json_dir):
    # Собираем все подпапки (исключая analytics)
    subdirs = [d for d in os.listdir(json_dir)
               if os.path.isdir(os.path.join(json_dir, d)) and d != "analytics"]

    # Если нет подпапок, анализируем только корневую директорию
    if not subdirs:
        subdirs = [""]

    # Для сбора общей статистики
    total_confidences = []
    total_recognized = 0
    total_unrecognized = 0

    for subdir in subdirs:
        current_dir = os.path.join(json_dir, subdir)
        folder_name = subdir if subdir else "all_data"

        # Анализируем текущую папку
        confidences, recognized, unrecognized = analyze_folder(current_dir, folder_name)

        # Добавляем к общей статистике
        total_confidences.extend(confidences)
        total_recognized += recognized
        total_unrecognized += unrecognized

        # Сохраняем графики для текущей папки
        if confidences:
            conf_output_path = os.path.join(OUTPUT_DIR, "confidence_histograms", f"{folder_name}.png")
            plot_confidence_histogram(confidences, conf_output_path, folder_name)

        recog_output_path = os.path.join(OUTPUT_DIR, "recognition_status", f"{folder_name}.png")
        plot_recognition_status(recognized, unrecognized, recog_output_path, folder_name)

    # Сохраняем общую статистику
    if total_confidences:
        conf_output_path = os.path.join(OUTPUT_DIR, "confidence_histograms", "TOTAL.png")
        plot_confidence_histogram(total_confidences, conf_output_path, "TOTAL")

    recog_output_path = os.path.join(OUTPUT_DIR, "recognition_status", "TOTAL.png")
    plot_recognition_status(total_recognized, total_unrecognized, recog_output_path, "TOTAL")

    print(f"Графики успешно созданы и сохранены в {OUTPUT_DIR}")


if __name__ == "__main__":
    analyze_and_plot(JSON_DIR)