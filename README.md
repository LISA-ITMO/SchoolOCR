# SchoolOCR

[![Python 3.11.7](https://img.shields.io/badge/Python-3.11.7-blue?logo=python)](https://www.python.org/)
[![OpenCV 4.11.0](https://img.shields.io/badge/OpenCV-4.11.0-blue)](https://opencv.org/)
[![TensorFlow 2.19.0](https://img.shields.io/badge/TensorFlow-2.19.0-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Ultralytics 8.3.105](https://img.shields.io/badge/Ultralytics-8.3.105-red)](https://ultralytics.com/)
[![FastAPI 0.115.12](https://img.shields.io/badge/FastAPI-0.115.12-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE.md)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://hub.docker.com/)

## Описание
**SchoolOCR** - API для автоматического распознавания данных с титульных листов Всероссийских Проверочных Работ (ВПР) с использованием нейросетевых технологий. Решение позволяет:

- Автоматизировать обработку бланков
- Снизить нагрузку на преподавателей
- Минимизировать ошибки ручного ввода

Пример титульной страницы:  
![Пример бланка](https://github.com/user-attachments/assets/37c95311-d113-4e8f-acbf-c6ce7ed68a10)

## Ключевые особенности
- **Распознавание структурированных данных**:
  - Предмет, класс, вариант
  - Код участника
  - Баллы по заданиям
- **Поддержка форматов**:
  - Изображения (JPG/PNG)
  - PDF-документы
- **Технологический стек**:
  - YOLOv11 для детекции ячеек
  - Кастомная CNN для распознавания цифр
  - Tesseract OCR для распознавания печатного текста
- **API**:
  - RESTful интерфейс
  - Поддержка base64-кодирования
  - Авторизация по API-ключу
## Установка
1. Клонируйте репозиторий:
```bash
git clone https://github.com/LISA-ITMO/SchoolOCR
cd SchoolOCR
```
2. Создайте и активируйте виртуальное окружение:
```bash
# Для Windows
python -m venv venv
venv\Scripts\activate

# Для Linux/macOS
python3 -m venv venv
source venv/bin/activate
```
3. Установите зависимости:
```bash
pip install -r requirements.txt
```
4. Запустите API
```bash
python ./app.py
```
5. Можете сделать тестовый запрос
```bash
python ./scripts/app_interaction/sender.py
```

## Установка с помощью Docker
1. Клонируйте репозиторий:
```bash
git clone https://github.com/LISA-ITMO/SchoolOCR
cd SchoolOCR
```
2. Поднимите контейнер
```bash
docker-compose up --build
```
3. Можете сделать тестовый запрос
```bash
python ./scripts/app_interaction/sender.py
```

## Авторизация API

### Текущая реализация
Авторизация реализована в простейшем виде через статичные API-ключи в файле `api_keys.json`.

### Создание файла API-ключей

1. Создайте файл `api_keys.json` в корне проекта:
```json
{
    "keys": [
        "ваш_уникальный_ключ_1",
        "ваш_уникальный_ключ_2"
    ]
}
```

## Пример авторизованного запроса

```python
import requests
import base64

# Кодируем изображение в base64
with open("test_image.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Формируем запрос
url = "http://localhost:8000/recognize"
headers = {
    "Authorization": "42d354f4b6e38ff95553137e49f724c9bc429399"  # Ваш API-ключ
}
payload = {
    "image_base64": encoded_image
}

# Отправляем запрос
response = requests.post(
    url,
    headers=headers,
    json=payload
)

# Обрабатываем ответ
if response.status_code == 200:
    print("Успешный ответ:")
    print(response.json())
else:
    print(f"Ошибка {response.status_code}:")
    print(response.text)
```

## Краткая документация

### Основные модули

- `app.py`  
Точка входа FastAPI с эндпоинтами:

  `/recognize` - Основной обработчик изображений  
  `/healthcheck` - Проверка состояния сервиса  

- `utils/code_rec.py`
```python
def recognize_code(image: np.ndarray, model) -> str:
    """Извлекает код участника из изображения"""
```

- `utils/table_rec.py`
```python
def recognize_table(image: np.ndarray, model) -> dict:
    """Обрабатывает таблицу с ответами и возвращает баллы"""
```
- `utils/preprocess_general.py`
```python
  def preprocess_general(img):
    """Предобработка изображения перед распознаванием"""
```

### Модели

| Модель | Назначение | Путь |
|--------|------------|------|
| YOLOv11 | Детекция ячеек | `models/cell_detect.pt` |
| CNN | Распознавание цифр | `models/digit_model.h5` |

## Основные зависимости

### Ключевые фреймворки
| Библиотека | Версия | Назначение |
|------------|--------|------------|
| `fastapi` | 0.115.12 | Веб-фреймворк для API |
| `uvicorn` | 0.34.0 | ASGI-сервер |
| `tensorflow` | 2.19.0 | Нейросетевой фреймворк |
| `ultralytics` | 8.3.105 | YOLO модели |

### Обработка изображений
| Библиотека | Версия |
|------------|--------|
| `opencv-python` | 4.11.0 |
| `pillow` | 11.1.0 |
| `pytesseract` | 0.3.13 |
| `scikit-image` | 0.25.2 |

### Дополнительные
| Библиотека | Назначение |
|------------|------------|
| `PyMuPDF` | Работа с PDF |
| `pydantic` | Валидация данных |
| `numpy` | Матричные операции |

[Полный список зависимостей →](requirements.txt)

## Требования к скану
В настоящий момент решение крайне чувствительно к исходному качеству, поэтому для наиболее корректной обработки важно соблюдать требования:
- Скан, насколько это возможно, сделать ровным и четким;
- Цифры пишутся раздельно, черной (лучше гелиевой) ручкой, без разъединений - в идеале, как печатные. Для таблиц - ровно в клетке, не выходя за рамки;
- Для каждой цифры в таблице - отдельная клетка, границы должны быть как можно четче.
- Передавать API файл в jpg или pdf формате через base64, разрешение исходное

## Конференции
[Тезис КМУ](https://kmu.itmo.ru/digests/article/15643)
