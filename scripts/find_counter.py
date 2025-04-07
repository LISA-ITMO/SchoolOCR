import cv2
import numpy as np
from utils.preprocess_general import preprocess_general

# Загрузка изображения
img = cv2.imread('debug_tables/table_page_1.png')
preprocessed = preprocess_general(img)

# Дилатация
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(preprocessed, kernel, iterations=2)

# Поиск контуров
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Сортировка контуров по площади
contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:2]  # Берем два самых больших

# Список для хранения вырезанных таблиц
cropped_tables = []

for i, contour in enumerate(contours_sorted):
    # Получаем координаты ограничивающего прямоугольника
    x, y, w, h = cv2.boundingRect(contour)

    padding = 0
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, img.shape[1])  # Правая граница
    y2 = min(y + h + padding, img.shape[0])  # Нижняя граница

    # Вырезаем таблицу с отступом
    cropped_table = img[y1:y2, x1:x2]
    cropped_tables.append(cropped_table)

    # Для отладки: покажем вырезанные таблицы
    cv2.imshow(f'Table {i + 1}', cropped_table)
    cv2.waitKey(0)

cv2.destroyAllWindows()

cv2.imwrite('./debug_tables/cropped.jpg', cropped_tables[0])