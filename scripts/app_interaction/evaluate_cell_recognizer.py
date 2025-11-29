import os
import sys
import cv2
import numpy as np
from collections import defaultdict
import json

# Добавляем путь для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.cell_recognizer import CellRecognizer


def evaluate_cell_recognition(data_dir, output_report_path=None):
    # Создаем распознаватель
    print("Инициализация CellRecognizer...")
    recognizer = CellRecognizer(debug=False)
    print("CellRecognizer инициализирован")

    # Собираем статистику
    stats = {
        'total_images': 0,
        'correct_predictions': 0,
        'incorrect_predictions': 0,
        'accuracy': 0.0,
        'per_digit_stats': {},
        'confusion_matrix': defaultdict(lambda: defaultdict(int)),
        'detailed_results': []
    }

    # Проходим по всем папкам с цифрами
    for digit_dir in sorted(os.listdir(data_dir)):
        digit_path = os.path.join(data_dir, digit_dir)

        if not os.path.isdir(digit_path):
            continue

        try:
            expected_digit = int(digit_dir) if digit_dir.isdigit() else None
            if expected_digit is None and digit_dir.lower() != 'x':
                continue

            # Для папки 'x' ожидаем цифру 11
            if digit_dir.lower() == 'x':
                expected_digit = 11

            print(f"\nОбрабатываем папку: {digit_dir} (ожидаемая цифра: {expected_digit})")

            # Инициализируем статистику для этой цифры
            if expected_digit not in stats['per_digit_stats']:
                stats['per_digit_stats'][expected_digit] = {
                    'total': 0,
                    'correct': 0,
                    'accuracy': 0.0,
                    'common_errors': defaultdict(int)
                }

            digit_stats = stats['per_digit_stats'][expected_digit]

            # Обрабатываем все изображения в папке
            image_files = [f for f in os.listdir(digit_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for image_file in image_files:
                image_path = os.path.join(digit_path, image_file)

                try:
                    # Загружаем изображение
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"  Не удалось загрузить: {image_file}")
                        continue

                    # Распознаем цифру
                    predicted_digit, probability, prob_all = recognizer.recognize_cell(
                        image, image_file
                    )

                    # Обновляем статистику
                    stats['total_images'] += 1
                    digit_stats['total'] += 1

                    is_correct = (predicted_digit == expected_digit)

                    if is_correct:
                        stats['correct_predictions'] += 1
                        digit_stats['correct'] += 1
                    else:
                        stats['incorrect_predictions'] += 1
                        digit_stats['common_errors'][predicted_digit] += 1

                    # Обновляем матрицу ошибок
                    stats['confusion_matrix'][expected_digit][predicted_digit] += 1

                    # Сохраняем детальный результат
                    stats['detailed_results'].append({
                        'file': image_file,
                        'expected': expected_digit,
                        'predicted': predicted_digit,
                        'probability': probability,
                        'correct': is_correct,
                        'all_probabilities': prob_all
                    })

                    status = "✓" if is_correct else f"✗ (распознано как {predicted_digit})"
                    print(f"  {image_file}: {status} (вероятность: {probability:.3f})")

                except Exception as e:
                    print(f"  Ошибка обработки {image_file}: {e}")
                    continue

            # Вычисляем точность для этой цифры
            if digit_stats['total'] > 0:
                digit_stats['accuracy'] = digit_stats['correct'] / digit_stats['total']

        except Exception as e:
            print(f"Ошибка обработки папки {digit_dir}: {e}")
            continue

    # Вычисляем общую точность
    if stats['total_images'] > 0:
        stats['accuracy'] = stats['correct_predictions'] / stats['total_images']

    # Формируем отчет
    report = generate_report(stats)

    # Сохраняем отчет если указан путь
    if output_report_path:
        save_report(report, stats, output_report_path)

    return stats, report


def generate_report(stats):
    """Генерирует текстовый отчет"""
    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append("ОТЧЕТ О ТОЧНОСТИ РАСПОЗНАВАНИЯ ЯЧЕЕК")
    report_lines.append("=" * 60)

    report_lines.append(f"\nОБЩАЯ СТАТИСТИКА:")
    report_lines.append(f"Всего изображений: {stats['total_images']}")
    report_lines.append(f"Правильно распознано: {stats['correct_predictions']}")
    report_lines.append(f"Неправильно распознано: {stats['incorrect_predictions']}")
    report_lines.append(f"Точность: {stats['accuracy']:.4f} ({stats['accuracy'] * 100:.2f}%)")

    report_lines.append(f"\nСТАТИСТИКА ПО ЦИФРАМ:")
    report_lines.append("-" * 40)

    for digit in sorted(stats['per_digit_stats'].keys()):
        digit_stat = stats['per_digit_stats'][digit]
        if digit_stat['total'] > 0:
            digit_name = 'x' if digit == 11 else str(digit)
            report_lines.append(
                f"Цифра {digit_name}: {digit_stat['correct']}/{digit_stat['total']} "
                f"({digit_stat['accuracy']:.4f} | {digit_stat['accuracy'] * 100:.2f}%)"
            )

            # Показываем частые ошибки
            if digit_stat['common_errors']:
                errors_str = ", ".join([f"{k}({v})" for k, v in
                                        sorted(digit_stat['common_errors'].items(),
                                               key=lambda x: x[1], reverse=True)[:3]])
                report_lines.append(f"  Частые ошибки: {errors_str}")

    report_lines.append(f"\nМАТРИЦА ОШИБОК (ожидаемый → распознанный):")
    report_lines.append("-" * 50)

    # Заголовок матрицы
    digits = sorted(set(stats['confusion_matrix'].keys()) |
                    set([k for v in stats['confusion_matrix'].values() for k in v.keys()]))
    header = "     " + "".join([f"{('x' if d == 11 else str(d)):>4}" for d in digits])
    report_lines.append(header)

    for expected in digits:
        row = f"{('x' if expected == 11 else str(expected)):>3} |"
        for predicted in digits:
            count = stats['confusion_matrix'][expected][predicted]
            marker = "  * " if expected == predicted and count > 0 else f"{count:>4}"
            row += marker
        report_lines.append(row)

    report_lines.append(f"\nПРИМЕРЫ ОШИБОК:")
    report_lines.append("-" * 30)

    errors = [r for r in stats['detailed_results'] if not r['correct']]
    for error in errors[:10]:  # Показываем первые 10 ошибок
        expected_name = 'x' if error['expected'] == 11 else str(error['expected'])
        predicted_name = 'x' if error['predicted'] == 11 else str(error['predicted'])
        report_lines.append(
            f"{error['file']}: ожидалось {expected_name}, распознано {predicted_name} "
            f"(вероятность: {error['probability']:.3f})"
        )

    if len(errors) > 10:
        report_lines.append(f"... и еще {len(errors) - 10} ошибок")

    return "\n".join(report_lines)


def save_report(report_text, stats, output_path):
    """Сохраняет отчет в файл"""
    try:
        # Сохраняем текстовый отчет
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\nТекстовый отчет сохранен: {output_path}")

        # Сохраняем детальную статистику в JSON
        json_path = output_path.replace('.txt', '_detailed.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        print(f"Детальный отчет (JSON) сохранен: {json_path}")

    except Exception as e:
        print(f"Ошибка сохранения отчета: {e}")


def main():
    data_dir = "data/organized_cells"
    output_report_path = "cell_recognition_report.txt"

    if not os.path.exists(data_dir):
        print(f"Ошибка: папка {data_dir} не существует")
        return

    print("Начинаем оценку точности распознавания...")
    print(f"Данные: {data_dir}")
    print(f"Отчет: {output_report_path}")

    stats, report = evaluate_cell_recognition(data_dir, output_report_path)

    print("\n" + "=" * 60)
    print("ОЦЕНКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print(report)


if __name__ == '__main__':
    main()