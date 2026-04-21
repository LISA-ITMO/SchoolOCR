"""
Скрипт сбора статистики распознавания для /recognize.

Конфигурация — секция CONFIG ниже.
Результаты сохраняются в OUTPUT_DIR:
  results/  — по одному JSON-файлу на каждый отправленный файл
  stats.json — сводная статистика
  stats.png  — визуализация через OpenCV
"""

import json
import time
import mimetypes
from pathlib import Path

import cv2
import numpy as np
import requests

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SERVER_URL = "http://localhost:8000/recognize"
API_KEY    = "42d354f4b6e38ff95553137e49f724c9bc429399"

BASE_DIR = Path(__file__).parent.parent / "check_school"

# Папки с данными (можно добавлять / убирать)
DATA_FOLDERS = [
    BASE_DIR / "Okasana",
]

OUTPUT_DIR = Path(__file__).parent / "recognition_stats_output"

# Таймаут ожидания одного файла (секунд)
REQUEST_TIMEOUT = 300

# ──────────────────────────────────────────────────────────────────────────────


def collect_files(folders: list[Path]) -> list[Path]:
    exts = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
    files: list[Path] = []
    for folder in folders:
        if not folder.exists():
            print(f"[WARN] Папка не найдена: {folder}")
            continue
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    return files


def recognize_file(path: Path) -> dict | None:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    headers = {"X-API-Key": API_KEY}
    try:
        with open(path, "rb") as fh:
            resp = requests.post(
                SERVER_URL,
                files={"file": (path.name, fh, mime)},
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        if resp.status_code == 200:
            return {"status": "ok", "items": resp.json()}
        return {"status": "error", "http_code": resp.status_code, "detail": resp.text}
    except Exception as exc:
        return {"status": "exception", "detail": str(exc)}


def result_path(source_path: Path, results_dir: Path) -> Path:
    """
    Сохраняем зеркальную структуру папок внутри results_dir,
    чтобы файлы с одинаковым именем из разных папок не конфликтовали.
    Пример: Nailya_splitted/РЯ_5_1/page_1.pdf → results/Nailya_splitted/РЯ_5_1/page_1.json
    """
    # Ищем общий корень среди DATA_FOLDERS
    for folder in DATA_FOLDERS:
        try:
            rel = source_path.relative_to(folder)
            mirror = results_dir / folder.name / rel.parent / (rel.stem + ".json")
            return mirror
        except ValueError:
            continue
    # Fallback: просто относительно BASE_DIR
    try:
        rel = source_path.relative_to(BASE_DIR)
        return results_dir / rel.parent / (rel.stem + ".json")
    except ValueError:
        return results_dir / (source_path.stem + ".json")


# ─── статистика ───────────────────────────────────────────────────────────────

def build_stats(all_results: list[dict]) -> dict:
    total = len(all_results)
    ok = sum(1 for r in all_results if r["api_status"] == "ok")
    failed = total - ok

    has_errors   = 0
    has_warnings = 0
    no_items     = 0
    subjects: dict[str, int] = {}
    score_sum = 0
    score_count = 0

    for r in all_results:
        if r["api_status"] != "ok":
            continue
        items = r.get("items", [])
        if not items:
            no_items += 1
            continue
        for item in items:
            errs  = item.get("errors") or []
            warns = item.get("warnings") or []
            if errs:
                has_errors += 1
            if warns:
                has_warnings += 1
            subj = item.get("subject") or "unknown"
            subjects[subj] = subjects.get(subj, 0) + 1
            ts = item.get("total_score")
            if ts is not None:
                score_sum   += ts
                score_count += 1

    return {
        "total_files": total,
        "api_ok": ok,
        "api_failed": failed,
        "items_with_errors": has_errors,
        "items_with_warnings": has_warnings,
        "items_empty_response": no_items,
        "avg_total_score": round(score_sum / score_count, 2) if score_count else None,
        "subjects": subjects,
    }


# ─── OpenCV визуализация ──────────────────────────────────────────────────────

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_BG         = (245, 245, 245)
_FG         = (30,  30,  30)
_GREEN      = (60, 180,  60)
_RED        = (60,  60, 210)
_ORANGE     = (30, 140, 220)
_BLUE       = (200,  80,  30)


def _put(img, text, x, y, scale=0.55, color=_FG, thickness=1):
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, value, max_value, color, label, count):
    bar_h = int(h * value / max(max_value, 1))
    cv2.rectangle(img, (x, y + h - bar_h), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), _FG, 1)
    _put(img, f"{count}", x + w // 2 - 10, y + h - bar_h - 6, color=color, thickness=2)
    _put(img, label, x, y + h + 18, scale=0.45, color=_FG)


def build_image(stats: dict) -> np.ndarray:
    W, H = 900, 560
    img  = np.full((H, W, 3), _BG, dtype=np.uint8)

    # Заголовок
    _put(img, "Recognition Statistics", 20, 35, scale=0.9, color=_FG, thickness=2)
    cv2.line(img, (20, 45), (W - 20, 45), _FG, 1)

    # ── Сводка ──────────────────────────────────────────────
    y0 = 75
    summary = [
        (f"Total files:         {stats['total_files']}",  _FG),
        (f"API ok:              {stats['api_ok']}",       _GREEN),
        (f"API failed:          {stats['api_failed']}",   _RED),
        (f"Items with errors:   {stats['items_with_errors']}",   _RED),
        (f"Items with warnings: {stats['items_with_warnings']}", _ORANGE),
        (f"Empty responses:     {stats['items_empty_response']}", _ORANGE),
        (f"Avg total score:     {stats['avg_total_score']}",  _BLUE),
    ]
    for i, (txt, col) in enumerate(summary):
        _put(img, txt, 30, y0 + i * 28, scale=0.6, color=col, thickness=1)

    # ── Столбчатая диаграмма: ok / failed / errors / warnings ──
    bar_y   = 90
    bar_h   = 160
    bar_w   = 70
    gap     = 30
    x_start = 400
    bars = [
        (stats["api_ok"],                _GREEN,  "API ok"),
        (stats["api_failed"],            _RED,    "API fail"),
        (stats["items_with_errors"],     _RED,    "Errors"),
        (stats["items_with_warnings"],   _ORANGE, "Warnings"),
        (stats["items_empty_response"],  _ORANGE, "Empty"),
    ]
    max_val = max((b[0] for b in bars), default=1)
    for i, (val, col, lbl) in enumerate(bars):
        bx = x_start + i * (bar_w + gap)
        draw_bar(img, bx, bar_y, bar_w, bar_h, val, max_val, col, lbl, val)

    cv2.line(img, (20, 295), (W - 20, 295), _FG, 1)

    # ── Распределение по предметам ──────────────────────────
    _put(img, "Subjects:", 30, 325, scale=0.65, color=_FG, thickness=1)

    subjects = stats.get("subjects", {})
    if subjects:
        sorted_subj = sorted(subjects.items(), key=lambda x: -x[1])
        bar_y2  = 345
        bar_h2  = 130
        bar_w2  = max(30, min(80, (W - 60) // max(len(sorted_subj), 1) - 10))
        max_s   = max(v for _, v in sorted_subj)
        palette = [_GREEN, _BLUE, _ORANGE, _RED, (120, 180, 200), (200, 120, 180)]
        for i, (subj, cnt) in enumerate(sorted_subj):
            bx  = 30 + i * (bar_w2 + 10)
            col = palette[i % len(palette)]
            draw_bar(img, bx, bar_y2, bar_w2, bar_h2, cnt, max_s, col,
                     subj[:10], cnt)
            if bx + bar_w2 > W - 20:
                _put(img, f"...+{len(sorted_subj)-i-1} more", bx + 5, bar_y2 + bar_h2 // 2,
                     scale=0.4, color=_FG)
                break

    # ── Процент успеха ──────────────────────────────────────
    total = stats["total_files"]
    pct   = round(100 * stats["api_ok"] / total, 1) if total else 0
    _put(img, f"Success rate: {pct}%", 30, H - 20, scale=0.65,
         color=_GREEN if pct >= 80 else _RED, thickness=1)

    return img


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(DATA_FOLDERS)
    print(f"Найдено файлов: {len(files)}")

    all_results: list[dict] = []

    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {path.relative_to(BASE_DIR.parent)}", end=" ... ", flush=True)

        out_path = result_path(path, results_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Если результат уже есть — не отправляем повторно
        if out_path.exists():
            with open(out_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            all_results.append(saved)
            print("cached")
            continue

        t0  = time.time()
        res = recognize_file(path)
        elapsed = round(time.time() - t0, 1)

        record = {
            "file": str(path),
            "api_status": res.get("status", "error"),
            "elapsed_s": elapsed,
            **{k: v for k, v in res.items() if k != "status"},
        }

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)

        all_results.append(record)
        status_str = res.get("status", "?")
        print(f"{status_str} ({elapsed}s)")

    # Сохраняем сводную статистику
    stats = build_stats(all_results)
    stats_json_path = OUTPUT_DIR / "stats.json"
    with open(stats_json_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(f"\nСтатистика → {stats_json_path}")

    # Визуализация OpenCV
    img = build_image(stats)
    stats_img_path = OUTPUT_DIR / "stats.png"
    cv2.imwrite(str(stats_img_path), img)
    print(f"Изображение → {stats_img_path}")

    # Краткий итог
    print("\n─── Итог ───────────────────────────────")
    for k, v in stats.items():
        if k != "subjects":
            print(f"  {k}: {v}")
    if stats["subjects"]:
        print("  subjects:")
        for subj, cnt in sorted(stats["subjects"].items(), key=lambda x: -x[1]):
            print(f"    {subj}: {cnt}")


if __name__ == "__main__":
    main()
