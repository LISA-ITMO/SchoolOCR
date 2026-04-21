"""
Сравнение кодов участников: experiment (эталон) vs новый прогон.
Сравнение позиционное — по каждой из 5 цифр кода.

Результат в OUTPUT_DIR:
  codes_comparison.json       — детали по каждому файлу
  codes_summary.json          — сводная статистика
  codes_heatmap_<subject>.png — тепловая карта: строки=файлы, столбцы=позиции цифр
"""

from pathlib import Path
import json
import cv2
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent / "recognition_stats_output" / "results" / "Okasana"

EXPERIMENT_DIR = BASE / "experiment_rus_chem"

NEW_DIRS = [
    BASE / "русс яз 4 кл",
    BASE / "химия 8 кл",
]

OUTPUT_DIR = Path(__file__).parent / "recognition_stats_output" / "comparison"

# ──────────────────────────────────────────────────────────────────────────────


def load_experiment(exp_dir: Path) -> dict[str, dict]:
    results = {}
    for subdir in sorted(exp_dir.iterdir()):
        if not subdir.is_dir():
            continue
        rj = subdir / "result.json"
        if rj.exists():
            with open(rj, encoding="utf-8") as fh:
                results[subdir.name] = json.load(fh)
    return results


def load_new(new_dirs: list[Path]) -> dict[str, dict]:
    results = {}
    for d in new_dirs:
        if not d.exists():
            print(f"[WARN] Папка не найдена: {d}")
            continue
        for jf in sorted(d.glob("*.json")):
            with open(jf, encoding="utf-8") as fh:
                raw = json.load(fh)
            if raw.get("api_status") != "ok":
                continue
            items = raw.get("items") or []
            if items:
                results[jf.stem] = items[0]
    return results


def get_subject(exp: dict) -> str:
    return (exp.get("subject") or "unknown").lower()


# ─── позиционное сравнение кода ──────────────────────────────────────────────

def compare_code(exp: dict, new: dict) -> dict:
    exp_code = str(exp.get("participant_code") or "").strip()
    new_code = str(new.get("participant_code") or "").strip()

    n = max(len(exp_code), len(new_code))
    positions = []
    for i in range(n):
        ec = exp_code[i] if i < len(exp_code) else None
        nc = new_code[i] if i < len(new_code) else None
        positions.append({
            "pos":     i + 1,
            "exp_dig": ec,
            "new_dig": nc,
            "match":   ec is not None and nc is not None and ec == nc,
        })

    compared  = sum(1 for p in positions if p["exp_dig"] and p["new_dig"])
    matched   = sum(1 for p in positions if p["match"])

    return {
        "exp_subject": get_subject(exp),
        "exp_code":    exp_code,
        "new_code":    new_code,
        "full_match":  exp_code == new_code,
        "n_digits":    n,
        "compared":    compared,
        "matched":     matched,
        "mismatched":  compared - matched,
        "positions":   positions,
    }


# ─── сводная статистика ───────────────────────────────────────────────────────

def build_summary(details: dict[str, dict]) -> dict:
    total      = len(details)
    full_match = sum(1 for d in details.values() if d["full_match"])
    total_cmp  = sum(d["compared"]  for d in details.values())
    total_mat  = sum(d["matched"]   for d in details.values())

    pos_errors: dict[int, int] = {}
    for d in details.values():
        for p in d["positions"]:
            if p["exp_dig"] and p["new_dig"] and not p["match"]:
                pos_errors[p["pos"]] = pos_errors.get(p["pos"], 0) + 1

    return {
        "total_files":              total,
        "full_code_match":          full_match,
        "full_code_match_pct":      round(100 * full_match / max(total, 1), 1),
        "total_digits_compared":    total_cmp,
        "total_digits_matched":     total_mat,
        "digit_accuracy_pct":       round(100 * total_mat / max(total_cmp, 1), 1),
        "errors_by_position":       dict(sorted(pos_errors.items())),
    }


# ─── OpenCV тепловая карта ────────────────────────────────────────────────────

_FONT     = cv2.FONT_HERSHEY_SIMPLEX
_BG       = (245, 245, 245)
_FG       = (20, 20, 20)
_MATCH    = (120, 210, 100)
_MISMATCH = (70,  70, 220)
_MISSING  = (180, 180, 180)
_HEADER   = (60,  60,  60)


def _put(img, text, x, y, scale=0.38, color=_FG, thickness=1):
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def build_heatmap(subject_files: list[tuple[str, dict]], title: str) -> np.ndarray:
    if not subject_files:
        return np.zeros((100, 400, 3), dtype=np.uint8)

    n_digits = max(d["n_digits"] for _, d in subject_files)
    n_files  = len(subject_files)

    CELL_W   = 52
    CELL_H   = 22
    LABEL_W  = 220
    HEADER_H = 50
    FOOTER_H = 60

    W = LABEL_W + n_digits * CELL_W + 20
    H = HEADER_H + n_files * CELL_H + FOOTER_H

    img = np.full((H, W, 3), _BG, dtype=np.uint8)

    _put(img, title, 10, 22, scale=0.52, color=_FG, thickness=1)
    cv2.line(img, (0, 30), (W, 30), _HEADER, 1)

    # Заголовки столбцов
    for col in range(n_digits):
        cx = LABEL_W + col * CELL_W + CELL_W // 2 - 8
        _put(img, f"dig {col+1}", cx - 4, HEADER_H - 6, scale=0.33, color=_HEADER)

    # Строки файлов
    for row, (name, d) in enumerate(subject_files):
        y_top = HEADER_H + row * CELL_H
        short = name[-28:] if len(name) > 28 else name
        _put(img, short, 4, y_top + CELL_H - 6, scale=0.33, color=_FG)

        for col, p in enumerate(d["positions"]):
            x_left = LABEL_W + col * CELL_W
            ec, nc = p["exp_dig"], p["new_dig"]

            if ec is None or nc is None:
                color = _MISSING
                txt   = f"{ec or '?'}"
            elif p["match"]:
                color = _MATCH
                txt   = ec
            else:
                color = _MISMATCH
                txt   = f"{ec}/{nc}"

            cv2.rectangle(img,
                          (x_left + 1, y_top + 1),
                          (x_left + CELL_W - 1, y_top + CELL_H - 1),
                          color, -1)
            _put(img, txt, x_left + 4, y_top + CELL_H - 6, scale=0.35,
                 color=(255, 255, 255) if color != _MISSING else _FG)

        cv2.line(img, (0, y_top + CELL_H), (W, y_top + CELL_H), (210, 210, 210), 1)

    # Нижняя строка — ошибки по позиции
    bottom_y = HEADER_H + n_files * CELL_H + 18
    _put(img, "Errors:", 4, bottom_y, scale=0.35, color=_FG)
    for col in range(n_digits):
        x_left = LABEL_W + col * CELL_W
        err_count = sum(
            1 for _, d in subject_files
            for p in d["positions"]
            if p["pos"] == col + 1
            and p["exp_dig"] and p["new_dig"] and not p["match"]
        )
        if err_count:
            cv2.rectangle(img,
                          (x_left + 1, bottom_y - 14),
                          (x_left + CELL_W - 1, bottom_y + 4),
                          _MISMATCH, -1)
            _put(img, str(err_count), x_left + 4, bottom_y + 2,
                 scale=0.3, color=(255, 255, 255))

    # Легенда
    leg_y = bottom_y + 22
    for color, label in [(_MATCH, "match"), (_MISMATCH, "mismatch  (exp/new)"), (_MISSING, "missing")]:
        cv2.rectangle(img, (4, leg_y - 8), (18, leg_y + 2), color, -1)
        _put(img, label, 22, leg_y, scale=0.33, color=_FG)
        leg_y += 14

    return img


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка experiment...")
    exp_data = load_experiment(EXPERIMENT_DIR)
    print(f"  {len(exp_data)} файлов")

    print("Загрузка new runs...")
    new_data = load_new(NEW_DIRS)
    print(f"  {len(new_data)} файлов")

    common   = sorted(set(exp_data) & set(new_data))
    only_exp = sorted(set(exp_data) - set(new_data))
    only_new = sorted(set(new_data) - set(exp_data))
    print(f"\nСовпадений: {len(common)}, только в exp: {len(only_exp)}, только в new: {len(only_new)}")

    details: dict[str, dict] = {}
    for name in common:
        details[name] = compare_code(exp_data[name], new_data[name])

    summary = build_summary(details)

    with open(OUTPUT_DIR / "codes_comparison.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "details": details}, fh, ensure_ascii=False, indent=2)
    print(f"Детали   → {OUTPUT_DIR / 'codes_comparison.json'}")

    with open(OUTPUT_DIR / "codes_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"Сводка   → {OUTPUT_DIR / 'codes_summary.json'}")

    # Тепловые карты по предметам
    subjects: dict[str, list[tuple[str, dict]]] = {}
    for name, d in details.items():
        subjects.setdefault(d["exp_subject"], []).append((name, d))

    for subj, files in subjects.items():
        files_sorted = sorted(files, key=lambda x: x[0])
        img = build_heatmap(
            files_sorted,
            f"Participant codes: {subj}  (green=match, red=mismatch exp/new)",
        )
        img_path = OUTPUT_DIR / f"codes_heatmap_{subj}.png"
        cv2.imwrite(str(img_path), img)
        print(f"Карта    → {img_path}")

    # Консольный итог
    print("\n─── Итог ──────────────────────────────────────────────")
    print(f"  Файлов:               {summary['total_files']}")
    print(f"  Полное совпадение:    {summary['full_code_match']}  ({summary['full_code_match_pct']}%)")
    print(f"  Точность по цифрам:   {summary['total_digits_matched']}/{summary['total_digits_compared']}"
          f"  ({summary['digit_accuracy_pct']}%)")
    if summary["errors_by_position"]:
        print("  Ошибок по позиции цифры:")
        for pos, cnt in summary["errors_by_position"].items():
            print(f"    pos {pos}: {cnt}")

    # Все расхождения
    diffs = [(n, d) for n, d in sorted(details.items()) if not d["full_match"]]
    if diffs:
        print("\n─── Все расхождения ────────────────────────────────────")
        for name, d in diffs:
            print(f"  {name:40s}  exp={d['exp_code']!r:10s}  new={d['new_code']!r:10s}"
                  f"  mismatched_digits={d['mismatched']}")


if __name__ == "__main__":
    main()
