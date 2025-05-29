import os
import cv2
import numpy as np
import tensorflow as tf
import shutil
from utils.mnist_preprocess_cell import preprocess_image

# Загрузка модели
model = tf.keras.models.load_model(
    "./mnist_with_x_model.keras"
)  # Убедитесь, что модель в той же папке
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "11"]

# Пути к папкам
input_folder = "../debug_cells/original"  # Папка с исходными изображениями
output_base = "./classified_cells_X"  # Папка для классифицированных изображений

# Создаем папки классов, если их нет
os.makedirs(output_base, exist_ok=True)
for class_name in classes:
    os.makedirs(os.path.join(output_base, class_name), exist_ok=True)


# Функция для обработки и классификации изображений
def process_and_classify_images():
    """
    Processes images from a folder, classifies them using a given model, and moves them to class-specific subfolders.

        Args:
            input_folder: The path to the directory containing the images to be processed.
            model: The pre-trained model used for image classification.
            classes: A list of class names corresponding to the model's output indices.
            output_base: The base directory where class-specific subfolders will be created and images moved into.

        Returns:
            None: This function performs file system operations and prints status messages; it does not return a value.
    """
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            try:
                # Чтение изображения
                img_path = os.path.join(input_folder, filename)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"Не удалось прочитать изображение: {filename}")
                    continue

                # Предварительная обработка
                processed_img, _ = preprocess_image(image)

                if processed_img is None:
                    print(f"Не удалось обработать изображение: {filename}")
                    continue

                # Классификация
                predictions = model.predict(processed_img)
                predicted_class = np.argmax(predictions[0])
                class_name = classes[predicted_class]

                # Перемещение файла
                output_path = os.path.join(output_base, class_name, filename)
                shutil.move(img_path, output_path)

                print(f"Изображение {filename} классифицировано как {class_name}")

            except Exception as e:
                print(f"Ошибка при обработке {filename}: {str(e)}")


# Запуск обработки
process_and_classify_images()
print("Классификация завершена!")
