FROM python:3.11.7-slim-bookworm

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    wget \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Скачивание лучших языковых данных (tessdata_best)
RUN mkdir -p /usr/share/tesseract-ocr/tessdata && \
    wget -O /usr/share/tesseract-ocr/tessdata/rus.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata && \
    wget -O /usr/share/tesseract-ocr/tessdata/osd.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/osd.traineddata

WORKDIR /app
COPY requirements.txt .
COPY app.py .
COPY config.json .
COPY *.keras .
COPY *.h5 .
COPY services/ ./services/

# Установка Python-зависимостей
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Оптимизированные переменные окружения
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata \
    OMP_THREAD_LIMIT=1 \
    PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]