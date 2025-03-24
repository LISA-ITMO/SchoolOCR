FROM python:3.11.7-slim-bookworm

# Обновление пакетов и установка зависимостей
RUN apt-get update && apt-get install -y \
    wget \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    libgl1-mesa-glx \
    libtiff5-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libwebp-dev \
    libglib2.0-dev \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Указываем путь к Tesseract data
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata

# Загрузка моделей Tesseract
RUN mkdir -p /usr/share/tesseract-ocr/tessdata && \
    wget -O /usr/share/tesseract-ocr/tessdata/rus.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata && \
    wget -O /usr/share/tesseract-ocr/tessdata/osd.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/osd.traineddata

# Оптимизация многопоточности
ENV OMP_THREAD_LIMIT=2

# Настройка рабочего каталога
WORKDIR /app

# Копирование файлов проекта
COPY requirements.txt .
COPY app.py .
COPY config.json .
COPY *.keras .
COPY *.h5 .
COPY services/ ./services/

# Установка Python-зависимостей
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Запуск приложения
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
