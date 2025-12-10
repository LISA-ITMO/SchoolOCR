FROM python:3.11.7-slim-bookworm

ARG BUILD_VERSION=unknown
ENV APP_VERSION=$BUILD_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
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
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata
RUN mkdir -p "$TESSDATA_PREFIX" && \
    wget -q -O "$TESSDATA_PREFIX/rus.traineddata" https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata && \
    wget -q -O "$TESSDATA_PREFIX/osd.traineddata" https://github.com/tesseract-ocr/tessdata_best/raw/main/osd.traineddata

ENV OMP_THREAD_LIMIT=2

RUN useradd -ms /bin/bash appuser

WORKDIR /app

COPY requirements-base.txt .
COPY requirements-ml.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements-base.txt

RUN pip install -r requirements-ml.txt

COPY app/ ./app

RUN chown -R appuser:appuser /app

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
