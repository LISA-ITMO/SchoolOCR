# SchoolOCR

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)

REST API для автоматического распознавания титульных листов ВПР: предмет, класс, вариант, код участника, баллы по заданиям.

Поддерживаемые форматы входных файлов: **JPEG, PNG, PDF**.

![Пример бланка](https://github.com/user-attachments/assets/37c95311-d113-4e8f-acbf-c6ce7ed68a10)

---

## Быстрый старт (Docker)

```bash
git clone -b master-lite https://github.com/LISA-ITMO/SchoolOCR
cd SchoolOCR
docker compose up --build -d
```

Сервис поднимается на порту `8000`.

### Опциональный Ollama (для LLM-распознавания)

Для использования `/llm/recognize` с локальной моделью нужен GPU с поддержкой NVIDIA Container Toolkit:

```bash
docker compose --profile ollama up --build -d
```

---

## Локальная установка

**Требования:** Python 3.11, Tesseract OCR с русским языком.

```bash
# Tesseract (Ubuntu/Debian)
apt install tesseract-ocr tesseract-ocr-rus

# Tesseract (macOS)
brew install tesseract tesseract-lang
```

```bash
git clone -b master-lite https://github.com/LISA-ITMO/SchoolOCR
cd SchoolOCR

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Авторизация

Все эндпоинты защищены API-ключом, передаётся в заголовке:

```
X-API-Key: <ключ>
```

Ключи хранятся в `app/api_keys.json`. Файл читается при старте — после изменения нужен перезапуск.

```json
{
  "keys": ["ваш_ключ_1", "ваш_ключ_2"]
}
```

---

## API

### `POST /recognize`

Основное распознавание. Принимает файл, возвращает массив результатов (по одному элементу на страницу).

**Запрос:**

```bash
curl -X POST http://localhost:8000/recognize \
  -H "X-API-Key: ваш_ключ" \
  -F "file=@bланк.jpg"
```

```python
import requests

resp = requests.post(
    "http://localhost:8000/recognize",
    headers={"X-API-Key": "ваш_ключ"},
    files={"file": open("бланк.jpg", "rb")},
)
print(resp.json())
```

**Ответ:**

```json
[
  {
    "id": "uuid",
    "subject": "Математика",
    "grade": "5",
    "variant": "1",
    "participant_code": "00123",
    "total_score": 14,
    "scores": {
      "1": [3, 0.98],
      "2": [2, 0.91],
      "3": ["x", 0.87]
    },
    "scores_details": {
      "1": {"0": 0.01, "1": 0.0, "2": 0.0, "3": 0.98, ...}
    },
    "errors": null,
    "warnings": ["Низкая уверенность в заданиях: 5, 7"]
  }
]
```

Поле `scores`: `{ "номер_задания": [значение, вероятность] }`.  
Значение — целое число `0–9`, либо `"-"` или `"x"` (специальные символы).

---

### `POST /llm/recognize`

Экспериментальное распознавание через мультимодальную LLM (Ollama Cloud). Обрабатывает документ целиком одним запросом.

**Параметры формы:**

| Поле      | Тип    | Обязательный | Описание                                              |
|-----------|--------|:------------:|-------------------------------------------------------|
| `file`    | file   | да           | PDF или изображение                                   |
| `api_key` | string | да*          | API-ключ Ollama Cloud (`*` или в `config/api_keys.json`) |
| `model`   | string | нет          | Модель Ollama, по умолчанию `qwen3-vl:235b`           |
| `prompt`  | string | нет          | Кастомный промпт (по умолчанию берётся из конфига)    |

**Запрос:**

```bash
curl -X POST http://localhost:8000/llm/recognize \
  -H "X-API-Key: ваш_ключ" \
  -F "file=@бланк.pdf" \
  -F "api_key=ваш_ollama_cloud_ключ"
```

**Ответ** — JSON той же структуры, что и `/recognize`, но `scores` без вероятностей:

```json
{
  "subject": "Русский язык",
  "grade": "7",
  "variant": "2",
  "participant_code": "00456",
  "total_score": 18,
  "scores": {"1": 3, "2": 2, "3": "X"},
  "errors": null,
  "warnings": null
}
```

> **Важно:** API-ключ Ollama Cloud и API-ключ SchoolOCR — разные ключи. Ключ Ollama получается в личном кабинете [ollama.com](https://ollama.com).

---

## Конфигурация

### `app/recognizers/config_new.json`

Настройки алгоритма распознавания:

```json
{
  "default": {
    "recognition": {
      "use_llm": false,
      "confidence_threshold": 0.6,
      "llm_trigger_threshold": 0.6
    },
    "llm": {
      "api_url": "https://ollama.com/api/chat",
      "model": "qwen3-vl:235b"
    }
  }
}
```

| Параметр               | Описание                                                    |
|------------------------|-------------------------------------------------------------|
| `use_llm`              | Использовать LLM как fallback при низкой уверенности CNN   |
| `confidence_threshold` | Порог уверенности, ниже которого задание помечается в `warnings` |
| `llm_trigger_threshold`| Порог уверенности CNN, ниже которого вызывается LLM        |

---

## Требования к качеству скана

- Ровный скан без перекосов, чёткий
- Цифры — раздельно, чёрной ручкой, в пределах клетки
- Каждая цифра в таблице — в отдельной клетке с чёткими границами
- Рекомендуемое разрешение: 300 DPI

---

## Структура проекта

```
app/
├── main.py                   # Точка входа FastAPI
├── api_keys.json             # API-ключи сервиса
├── routers/
│   ├── recognize.py          # POST /recognize
│   └── llm.py                # POST /llm/recognize
├── recognizers/
│   ├── recognizer.py         # Оркестратор распознавания
│   ├── header_recognizer.py  # Tesseract: предмет, класс, вариант
│   ├── code_recognizer.py    # CNN: код участника
│   ├── table_recognizer.py   # YOLO + CNN: таблица баллов
│   ├── cell_recognizer.py    # CNN распознавание ячейки
│   └── config_manager.py     # Управление конфигурацией
├── ml/
│   └── loader.py             # Ленивая загрузка моделей
└── weights/
    ├── best_model_balanced.h5 # CNN модель (цифры)
    ├── cell_detect.pt         # YOLO модель (детекция ячеек)
    └── cell_detect_extra.pt   # YOLO fallback модель
```

---

## Публикации

[Тезис КМУ](https://kmu.itmo.ru/digests/article/15643)
