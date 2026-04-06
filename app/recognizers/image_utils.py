import cv2
import numpy as np
import fitz

def is_pdf(file_data: bytes) -> bool:
    return len(file_data) > 4 and file_data[:4] == b'%PDF'

def pdf_to_image(pdf_data: bytes) -> np.ndarray:
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    page = doc.load_page(0)
    zoom = 300 / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def decode_image(image_data: bytes) -> np.ndarray:
    if is_pdf(image_data):
        print("Обнаружен PDF файл. Конвертируем...")
        return pdf_to_image(image_data)
    else:
        print("Обнаружено изображение. Обрабатываем...")
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Неверный формат изображения")
        return resize_to_target(image)

def resize_to_target(image: np.ndarray, target_width: int = 2480, target_height: int = 3505) -> np.ndarray:
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

def extract_region(image: np.ndarray, coords: dict) -> np.ndarray:
    x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
    return image[y1:y2, x1:x2]
