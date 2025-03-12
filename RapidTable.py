import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lineless_table_rec.utils_table_recover import plot_rec_box_with_logic_info
from wired_table_rec import WiredTableRecognition
from mnist_preprocess_image import preprocess_image

# Путь к изображению таблицы
IMG_PATH = './cropped_tables/page_5.jpg'

# Загрузка модели MNIST
model = tf.keras.models.load_model("mnist_model.keras")
print("Модель успешно загружена.")

# Инициализация движка для распознавания таблиц
table_engine = WiredTableRecognition()

# Распознавание таблицы
html, elasp, polygons, logic_points, ocr_res = table_engine(IMG_PATH, need_ocr=False)

# Визуализация распознанных областей таблицы
print("Polygons:", polygons)
print("Logic Points:", logic_points)
plot_rec_box_with_logic_info(IMG_PATH, "./table_rec_box.jpg", logic_points, polygons)

# Загрузка изображения таблицы
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Изображение по пути {IMG_PATH} не найдено.")
print("Изображение успешно загружено.")

# Выделение нужных ячеек (вторая строка)
second_row_cells = []
for i, logic in enumerate(logic_points):
    if logic[0] == 1 and logic[1] == 1:  # Вторая строка
        second_row_cells.append(polygons[i])

# Убираем лишние ячейки (если нужно)
second_row_cells = second_row_cells[1:-2]

# Массив для хранения распознанных цифр
recognized_digits = []

# Создаем график для визуализации
plt.figure(figsize=(15, 5))

# Обработка каждой ячейки
for i, cell in enumerate(second_row_cells):
    x1, y1, x2, y2 = map(int, cell)
    cell_img = img[y1:y2, x1:x2]

    # Проверяем, что изображение ячейки не пустое
    if cell_img is None or cell_img.size == 0:
        print(f"Ячейка {i + 1}: Изображение пустое или невалидное.")
        continue

    # Предобработка изображения ячейки
    input_data, processed_img = preprocess_image(cell_img)
    if input_data is None:
        print(f"Ячейка {i + 1}: Не удалось обработать изображение.")
        continue

    # Распознавание цифры
    predictions = model.predict(input_data)
    predicted_digit = np.argmax(predictions)
    predicted_prob = np.max(predictions)

    # Сохранение распознанной цифры
    recognized_digits.append(predicted_digit)

    # Вывод результата
    print(f"Ячейка {i + 1}: Распознана цифра {predicted_digit} с вероятностью {predicted_prob:.4f}")

    # Визуализация оригинальной ячейки
    plt.subplot(2, len(second_row_cells), i + 1)
    plt.imshow(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original {i + 1}")
    plt.axis('off')

    # Визуализация обработанной ячейки
    plt.subplot(2, len(second_row_cells), i + 1 + len(second_row_cells))
    plt.imshow(processed_img, cmap='gray')
    plt.title(f"Processed {i + 1}\nPred: {predicted_digit}\nProb: {predicted_prob:.4f}")
    plt.axis('off')

# Отображение графиков
plt.tight_layout()
plt.show()