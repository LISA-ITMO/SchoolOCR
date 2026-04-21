"""
Сравнение результатов распознавания: experiment (эталон) vs новый прогон.

Сравнение ПОЗИЦИОННОЕ — по порядку задач, не по именам ключей,
т.к. схемы нумерации задач различаются ("9(1)" vs "9", "1.1" vs "1").

Результат в OUTPUT_DIR:
  comparison.json       — детали по каждому файлу (позиционные diff-ы)
  summary.json          — сводная статистика
  heatmap_<subject>.png — тепловая карта: строки=файлы, столбцы=позиции задач
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


# ─── загрузка ─────────────────────────────────────────────────────────────────

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


# ─── нормализация: scores → список значений по порядку ───────────────────────

def _to_int(v) -> int | None:
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def scores_to_list_exp(scores: dict) -> list[tuple[str, int | None]]:
    """[(key, value), ...]  в порядке вставки"""
    return [(k, _to_int(v)) for k, v in scores.items()]


def scores_to_list_new(scores: dict) -> list[tuple[str, int | None]]:
    """new format: {"1": [value, conf]}  →  [(key, value), ...]"""
    result = []
    for k, v in scores.items():
        if isinstance(v, (list, tuple)):
            result.append((k, _to_int(v[0])))
        else:
            result.append((k, _to_int(v)))
    return result


# ─── позиционное сравнение пары ──────────────────────────────────────────────

def compare_pair(exp: dict, new: dict) -> dict:
    exp_list = scores_to_list_exp(exp.get("scores") or {})
    new_list = scores_to_list_new(new.get("scores") or {})

    positions = []
    n = max(len(exp_list), len(new_list))
    for i in range(n):
        exp_key, exp_val = exp_list[i] if i < len(exp_list) else (None, None)
        new_key, new_val = new_list[i] if i < len(new_list) else (None, None)
        match = (exp_val is not None and new_val is not None and exp_val == new_val)
        positions.append({
            "pos":     i + 1,
            "exp_key": exp_key,
            "exp_val": exp_val,
            "new_key": new_key,
            "new_val": new_val,
            "match":   match,
        })

    matched  = sum(1 for p in positions if p["match"])
    compared = sum(1 for p in positions if p["exp_val"] is not None and p["new_val"] is not None)

    return {
        "exp_subject":  exp.get("subject"),
        "exp_total":    exp.get("total_score"),
        "new_total":    new.get("total_score"),
        "n_positions":  n,
        "matched":      matched,
        "compared":     compared,
        "mismatched":   compared - matched,
        "positions":    positions,
        "exp_errors":   exp.get("errors"),
        "new_errors":   new.get("errors"),
    }


# ─── сводная статистика ───────────────────────────────────────────────────────

def build_summary(details: dict[str, dict]) -> dict:
    total = len(details)
    perfect = sum(1 for d in details.values() if d["mismatched"] == 0 and d["compared"] > 0)
    total_matched = sum(d["matched"] for d in details.values())
    total_compared = sum(d["compared"] for d in details.values())

    # Ошибки по позиции
    pos_errors: dict[int, int] = {}
    for d in details.values():
        for p in d["positions"]:
            if p["exp_val"] is not None and p["new_val"] is not None and not p["match"]:
                pos = p["pos"]
                pos_errors[pos] = pos_errors.get(pos, 0) + 1

    return {
        "total_files": total,
        "perfect_match": perfect,
        "perfect_match_pct": round(100 * perfect / max(total, 1), 1),
        "total_positions_compared": total_compared,
        "total_positions_matched": total_matched,
        "overall_accuracy_pct": round(100 * total_matched / max(total_compared, 1), 1),
        "errors_by_position": dict(sorted(pos_errors.items())),
    }


# ─── OpenCV: тепловая карта ───────────────────────────────────────────────────

_FONT    = cv2.FONT_HERSHEY_SIMPLEX
_BG      = (245, 245, 245)
_FG      = (20, 20, 20)
_MATCH   = (120, 210, 100)   # зелёный
_MISMATCH= (70,  70, 220)    # красный (BGR)
_MISSING = (180, 180, 180)   # серый
_HEADER  = (60,  60,  60)


def _put(img, text, x, y, scale=0.38, color=_FG, thickness=1):
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def build_heatmap(subject_files: list[tuple[str, dict]], title: str) -> np.ndarray:
    if not subject_files:
        return np.zeros((100, 400, 3), dtype=np.uint8)

    n_tasks = max(d["n_positions"] for _, d in subject_files)
    n_files = len(subject_files)

    CELL_W  = 36
    CELL_H  = 22
    LABEL_W = 220
    HEADER_H = 50
    FOOTER_H = 40

    W = LABEL_W + n_tasks * CELL_W + 20
    H = HEADER_H + n_files * CELL_H + FOOTER_H + 30

    img = np.full((H, W, 3), _BG, dtype=np.uint8)

    # Заголовок
    _put(img, title, 10, 22, scale=0.55, color=_FG, thickness=1)
    cv2.line(img, (0, 30), (W, 30), _HEADER, 1)

    # Номера позиций (заголовки столбцов)
    for col in range(n_tasks):
        cx = LABEL_W + col * CELL_W + CELL_W // 2 - 8
        _put(img, str(col + 1), cx, HEADER_H - 6, scale=0.33, color=_HEADER)

    # Строки файлов
    for row, (name, d) in enumerate(subject_files):
        y_top = HEADER_H + row * CELL_H

        # Метка файла
        short = name[-28:] if len(name) > 28 else name
        _put(img, short, 4, y_top + CELL_H - 6, scale=0.33, color=_FG)

        # Ячейки позиций
        for col, p in enumerate(d["positions"]):
            x_left = LABEL_W + col * CELL_W
            ev, nv = p["exp_val"], p["new_val"]

            if ev is None or nv is None:
                color = _MISSING
            elif p["match"]:
                color = _MATCH
            else:
                color = _MISMATCH

            cv2.rectangle(img, (x_left + 1, y_top + 1),
                          (x_left + CELL_W - 1, y_top + CELL_H - 1), color, -1)

            # Значение эталона / нового (e/n)
            txt = f"{ev if ev is not None else '?'}"
            if not p["match"] and nv is not None:
                txt = f"{ev}/{nv}"
            _put(img, txt, x_left + 2, y_top + CELL_H - 6, scale=0.28,
                 color=(255, 255, 255) if color != _MISSING else _FG)

        # Граница строки
        cv2.line(img, (0, y_top + CELL_H), (W, y_top + CELL_H), (210, 210, 210), 1)

    # Нижняя строка — количество ошибок по позиции
    bottom_y = HEADER_H + n_files * CELL_H + 18
    _put(img, "Errors:", 4, bottom_y, scale=0.35, color=_FG)
    for col in range(n_tasks):
        x_left = LABEL_W + col * CELL_W
        err_count = sum(
            1 for _, d in subject_files
            for p in d["positions"]
            if p["pos"] == col + 1
            and p["exp_val"] is not None and p["new_val"] is not None
            and not p["match"]
        )
        if err_count:
            cv2.rectangle(img, (x_left + 1, bottom_y - 14),
                          (x_left + CELL_W - 1, bottom_y + 4), _MISMATCH, -1)
            _put(img, str(err_count), x_left + 4, bottom_y + 2,
                 scale=0.3, color=(255, 255, 255))

    # Легенда
    leg_y = bottom_y + 20
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
        details[name] = compare_pair(exp_data[name], new_data[name])

    summary = build_summary(details)

    # ── Сохраняем JSON ──────────────────────────────────────
    with open(OUTPUT_DIR / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "details": details}, fh, ensure_ascii=False, indent=2)
    print(f"Детали   → {OUTPUT_DIR / 'comparison.json'}")

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"Сводка   → {OUTPUT_DIR / 'summary.json'}")

    # ── Тепловые карты по предметам ─────────────────────────
    subjects: dict[str, list[tuple[str, dict]]] = {}
    for name, d in details.items():
        subj = (d.get("exp_subject") or "unknown").lower()
        subjects.setdefault(subj, []).append((name, d))

    for subj, files in subjects.items():
        files_sorted = sorted(files, key=lambda x: x[0])
        img = build_heatmap(files_sorted, f"Heatmap: {subj}  (green=match, red=mismatch exp/new)")
        img_path = OUTPUT_DIR / f"heatmap_{subj}.png"
        cv2.imwrite(str(img_path), img)
        print(f"Карта    → {img_path}")

    # ── Консольный итог ─────────────────────────────────────
    print("\n─── Итог ──────────────────────────────────────────────")
    print(f"  Файлов:              {summary['total_files']}")
    print(f"  Точное совпадение:   {summary['perfect_match']}  ({summary['perfect_match_pct']}%)")
    print(f"  Точность позиций:    {summary['total_positions_matched']}/{summary['total_positions_compared']}"
          f"  ({summary['overall_accuracy_pct']}%)")
    if summary["errors_by_position"]:
        print("  Ошибок по позиции:")
        for pos, cnt in summary["errors_by_position"].items():
            bar = "█" * cnt
            print(f"    pos {pos:>2}: {cnt:>3}  {bar}")

    # ── Все расхождения ─────────────────────────────────────
    all_diffs = sorted(details.items(), key=lambda x: x[0])
    has_any = any(d["mismatched"] > 0 for _, d in all_diffs)
    if has_any:
        print("\n─── Все расхождения ────────────────────────────────────")
        for name, d in all_diffs:
            if d["mismatched"] == 0:
                continue
            bad = [(p["pos"], p["exp_val"], p["new_val"])
                   for p in d["positions"] if not p["match"]
                   and p["exp_val"] is not None and p["new_val"] is not None]
            bad_str = "  ".join(f"p{pos}:{ev}→{nv}" for pos, ev, nv in bad)
            print(f"  {name:40s}  mismatches={d['mismatched']}  [{bad_str}]")


if __name__ == "__main__":
    main()
