import os
import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_json(json_file):
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        total_cells = data.get("total_cells", 0)
        recognized_cells = 0
        digit_counts = {}
        confidence_scores = []
        for cell in data.get("cells", []):
            content = cell.get("content")
            probability = cell.get("probability", 0.0)

            if content is not None:
                recognized_cells += 1
                digit_counts[content] = digit_counts.get(content, 0) + 1
                confidence_scores.append(probability)

        return {
            "file_name": os.path.basename(json_file),
            "recognized_cells": recognized_cells,
            "digit_counts": digit_counts,
            "confidence_scores": confidence_scores,
        }
    except Exception as e:
        print(f"Ошибка при анализе файла {json_file}: {e}")
        return None


if __name__ == "__main__":
    processed_dir = "./processed_tables_v2"

    stats = []

    for page_dir in os.listdir(processed_dir):
        page_path = os.path.join(processed_dir, page_dir)

        if os.path.isdir(page_path):
            json_file = os.path.join(page_path, f"{page_dir}.json")
            if os.path.exists(json_file):
                print(f"Анализ файла: {json_file}")
                result = analyze_json(json_file)

                if result is not None:
                    stats.append(result)

    if stats:
        recognized_cells_per_page = [s["recognized_cells"] for s in stats]

        all_digit_counts = {}
        all_confidence_scores = []

        for s in stats:
            for digit, count in s["digit_counts"].items():
                all_digit_counts[digit] = all_digit_counts.get(digit, 0) + count
            all_confidence_scores.extend(s["confidence_scores"])

        # === Гистограмма 1: Количество распознанных ячеек на странице ===
        plt.figure(figsize=(10, 5))
        bins = range(min(recognized_cells_per_page), max(recognized_cells_per_page) + 2)
        hist, bin_edges = np.histogram(recognized_cells_per_page, bins=bins)

        plt.bar(bin_edges[:-1], hist, align='edge', edgecolor='black', color='lightgreen', width=0.8)
        plt.title("Количество распознанных ячеек на странице")
        plt.xlabel("Количество распознанных ячеек")
        plt.ylabel("Количество страниц")
        plt.xticks(bin_edges[:-1])

        for i, count in enumerate(hist):
            if count > 0:
                plt.text(bin_edges[i], count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')

        histogram1_path = os.path.join(processed_dir, "recognized_cells_histogram.png")
        plt.savefig(histogram1_path)
        plt.close()
        print(f"Гистограмма количества распознанных ячеек сохранена как {histogram1_path}")

        # === Гистограмма 2: Частота встречаемости каждой цифры ===
        digits = list(all_digit_counts.keys())
        counts = list(all_digit_counts.values())

        plt.figure(figsize=(10, 5))
        bars = plt.bar(digits, counts, edgecolor='black', color='skyblue')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, str(height), ha='center', va='bottom', fontweight='bold')

        plt.title("Частота встречаемости цифр")
        plt.xlabel("Цифра")
        plt.ylabel("Количество")
        plt.xticks(range(10))  # Отображаем все цифры от 0 до 9
        histogram2_path = os.path.join(processed_dir, "digit_frequency_histogram.png")
        plt.savefig(histogram2_path)
        plt.close()
        print(f"Гистограмма частоты встречаемости цифр сохранена как {histogram2_path}")

        # === Гистограмма 3: Распределение уверенности ===
        if all_confidence_scores:
            plt.figure(figsize=(10, 5))
            bins = np.arange(0, 1.1, 0.1)  # Бины от 0 до 1 с шагом 0.1
            hist, bin_edges = np.histogram(all_confidence_scores, bins=bins)

            plt.bar(bin_edges[:-1], hist, align='edge', edgecolor='black', color='orange', width=0.08)
            plt.title("Распределение уверенности модели")
            plt.xlabel("Уверенность")
            plt.ylabel("Количество ячеек")
            plt.xticks(bin_edges)

            for i, count in enumerate(hist):
                if count > 0:
                    plt.text(bin_edges[i] + 0.04, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')

            histogram3_path = os.path.join(processed_dir, "confidence_histogram.png")
            plt.savefig(histogram3_path)
            plt.close()
            print(f"Гистограмма уверенности сохранена как {histogram3_path}")
        else:
            print("Нет данных для построения гистограммы уверенности.")

        output_stats_file = os.path.join(processed_dir, "statistics.json")
        with open(output_stats_file, "w") as f:
            json.dump({
                "pages_statistics": stats,
                "overall_digit_counts": all_digit_counts,
                "average_confidence": float(np.mean(all_confidence_scores)) if all_confidence_scores else 0.0,
            }, f, indent=4)

        print(f"\nСтатистика сохранена в файл: {output_stats_file}")
    else:
        print("Не найдено ни одного JSON-файла для анализа.")