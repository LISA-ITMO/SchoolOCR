import os
import sys
import cv2
from collections import defaultdict
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.cell_recognizer import CellRecognizer


def evaluate_cell_recognition(data_dir, output_report_path=None):
    print("Инициализация CellRecognizer...")
    recognizer = CellRecognizer(debug=False, enable_logging=False)
    print("CellRecognizer инициализирован")

    stats = {
        "total_images": 0,
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "accuracy": 0.0,
        "per_digit_stats": {},
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
        "detailed_results": [],
    }

    for digit_dir in sorted(os.listdir(data_dir)):
        digit_path = os.path.join(data_dir, digit_dir)

        if not os.path.isdir(digit_path):
            continue

        expected_digit = None
        if digit_dir.isdigit():
            expected_digit = int(digit_dir)
        elif digit_dir.lower() == "x":
            expected_digit = 11
        else:
            continue

        print(f"\nОбрабатываем папку: {digit_dir} (ожидаемая цифра: {expected_digit})")

        if expected_digit not in stats["per_digit_stats"]:
            stats["per_digit_stats"][expected_digit] = {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "common_errors": defaultdict(int),
            }

        digit_stats = stats["per_digit_stats"][expected_digit]

        image_files = [
            f
            for f in os.listdir(digit_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for image_file in image_files:
            image_path = os.path.join(digit_path, image_file)

            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  Не удалось загрузить: {image_file}")
                    continue

                res = recognizer.recognize_cell(image, image_file)
                predicted_digit = res.digit
                probability = float(res.prob or 0.0)
                prob_all = res.prob_all

                stats["total_images"] += 1
                digit_stats["total"] += 1

                is_correct = predicted_digit == expected_digit

                if is_correct:
                    stats["correct_predictions"] += 1
                    digit_stats["correct"] += 1
                else:
                    stats["incorrect_predictions"] += 1
                    digit_stats["common_errors"][predicted_digit] += 1

                stats["confusion_matrix"][expected_digit][predicted_digit] += 1

                stats["detailed_results"].append(
                    {
                        "file": image_file,
                        "expected": expected_digit,
                        "predicted": predicted_digit,
                        "probability": probability,
                        "correct": is_correct,
                        "all_probabilities": prob_all,
                        "llm_used": bool(res.llm_used),
                        "llm_digit": res.llm_digit,
                    }
                )

                status = "✓" if is_correct else f"✗ (распознано как {predicted_digit})"
                print(f"  {image_file}: {status} (вероятность: {probability:.3f})")

            except Exception as e:
                print(f"  Ошибка обработки {image_file}: {e}")
                continue

        if digit_stats["total"] > 0:
            digit_stats["accuracy"] = digit_stats["correct"] / digit_stats["total"]

    if stats["total_images"] > 0:
        stats["accuracy"] = stats["correct_predictions"] / stats["total_images"]

    report = generate_report(stats)

    if output_report_path:
        save_report(report, stats, output_report_path)

    return stats, report


def generate_report(stats):
    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append("ОТЧЕТ О ТОЧНОСТИ РАСПОЗНАВАНИЯ ЯЧЕЕК")
    report_lines.append("=" * 60)

    report_lines.append("\nОБЩАЯ СТАТИСТИКА:")
    report_lines.append(f"Всего изображений: {stats['total_images']}")
    report_lines.append(f"Правильно распознано: {stats['correct_predictions']}")
    report_lines.append(f"Неправильно распознано: {stats['incorrect_predictions']}")
    report_lines.append(f"Точность: {stats['accuracy']:.4f} ({stats['accuracy'] * 100:.2f}%)")

    report_lines.append("\nСТАТИСТИКА ПО ЦИФРАМ:")
    report_lines.append("-" * 40)

    for digit in sorted(stats["per_digit_stats"].keys()):
        digit_stat = stats["per_digit_stats"][digit]
        if digit_stat["total"] > 0:
            digit_name = "x" if digit == 11 else str(digit)
            report_lines.append(
                f"Цифра {digit_name}: {digit_stat['correct']}/{digit_stat['total']} "
                f"({digit_stat['accuracy']:.4f} | {digit_stat['accuracy'] * 100:.2f}%)"
            )

            if digit_stat["common_errors"]:
                errors_str = ", ".join(
                    [
                        f"{('x' if k == 11 else str(k))}({v})"
                        for k, v in sorted(
                            digit_stat["common_errors"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:3]
                    ]
                )
                report_lines.append(f"  Частые ошибки: {errors_str}")

    report_lines.append("\nМАТРИЦА ОШИБОК (ожидаемый → распознанный):")
    report_lines.append("-" * 50)

    digits = sorted(
        set(stats["confusion_matrix"].keys())
        | set(k for v in stats["confusion_matrix"].values() for k in v.keys())
    )

    header = "     " + "".join([f"{('x' if d == 11 else str(d)):>4}" for d in digits])
    report_lines.append(header)

    for expected in digits:
        row = f"{('x' if expected == 11 else str(expected)):>3} |"
        for predicted in digits:
            count = stats["confusion_matrix"][expected][predicted]
            row += ("  * " if expected == predicted and count > 0 else f"{count:>4}")
        report_lines.append(row)

    report_lines.append("\nПРИМЕРЫ ОШИБОК:")
    report_lines.append("-" * 30)

    errors = [r for r in stats["detailed_results"] if not r["correct"]]
    for error in errors[:10]:
        expected_name = "x" if error["expected"] == 11 else str(error["expected"])
        predicted = error["predicted"]
        predicted_name = "x" if predicted == 11 else str(predicted)
        report_lines.append(
            f"{error['file']}: ожидалось {expected_name}, распознано {predicted_name} "
            f"(вероятность: {error['probability']:.3f})"
        )

    if len(errors) > 10:
        report_lines.append(f"... и еще {len(errors) - 10} ошибок")

    return "\n".join(report_lines)


def save_report(report_text, stats, output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\nТекстовый отчет сохранен: {output_path}")

        json_path = output_path.replace(".txt", "_detailed.json")

        stats_for_json = dict(stats)
        stats_for_json["confusion_matrix"] = {
            str(k): {str(kk): vv for kk, vv in v.items()}
            for k, v in stats["confusion_matrix"].items()
        }
        stats_for_json["per_digit_stats"] = {
            str(k): {
                **{kk: vv for kk, vv in v.items() if kk != "common_errors"},
                "common_errors": {str(kk): vv for kk, vv in v["common_errors"].items()},
            }
            for k, v in stats["per_digit_stats"].items()
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats_for_json, f, ensure_ascii=False, indent=2)
        print(f"Детальный отчет (JSON) сохранен: {json_path}")

    except Exception as e:
        print(f"Ошибка сохранения отчета: {e}")


def main():
    data_dir = "data/organized_cells"
    output_report_path = "cell_recognition_report_new.txt"

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


if __name__ == "__main__":
    main()
