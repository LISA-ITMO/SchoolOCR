from PIL import Image
import os
import numpy as np


# Функция для сложения пикселей, близких к черному
def add_dark_pixels(images, threshold=50):
    """
    Складывает только пиксели, близкие к черному (ниже порога).

    :param images: Список изображений (PIL.Image)
    :param threshold: Пороговое значение для "близких к черному" пикселей (0–255)
    :return: Результирующее изображение (PIL.Image)
    """
    if not images:
        return None

    # Преобразуем изображения в массивы NumPy (grayscale)
    np_images = [np.array(img.convert("L")) for img in images]

    # Создаем пустой массив для результата
    result_array = np.zeros_like(np_images[0], dtype=np.uint32)

    # Складываем только пиксели, близкие к черному
    for img_array in np_images:
        mask = img_array < threshold  # Маска для пикселей ниже порога
        result_array += mask * img_array  # Складываем только эти пиксели

    # Нормализуем значения, чтобы они не превышали 255
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)

    # Преобразуем массив обратно в изображение
    result_image = Image.fromarray(result_array)

    return result_image


# Функция для сложения изображений из одной папки
def add_images_in_folder(folder_path, output_path, threshold=50):
    """
    Складывает все изображения из папки, учитывая только пиксели, близкие к черному.

    :param folder_path: Путь к папке с изображениями
    :param output_path: Путь для сохранения результата
    :param threshold: Пороговое значение для "близких к черному" пикселей (0–255)
    """
    images = []

    # Загружаем все изображения из папки
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path).convert("L")  # Конвертируем в grayscale
            images.append(img)

    if not images:
        print(f"В папке {folder_path} нет изображений.")
        return

    # Складываем изображения
    result_image = add_dark_pixels(images, threshold)

    # Сохраняем результат
    result_image.save(output_path)
    print(f"Результат сохранен в {output_path}")


# Функция для сложения всех объединенных изображений
def add_all_folders(root_folder, final_output_path, threshold=50):
    """
    Складывает все объединенные изображения из папок, учитывая только пиксели, близкие к черному.

    :param root_folder: Корневая папка с папками изображений
    :param final_output_path: Путь для сохранения финального результата
    :param threshold: Пороговое значение для "близких к черному" пикселей (0–255)
    """
    add_images_list = []  # Список для хранения изображений

    # Проходим по всем папкам в корневой папке
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            # Создаем временный файл для сложенных изображений из этой папки
            temp_output_path = os.path.join(root_folder, f"{folder_name}_added.png")
            add_images_in_folder(folder_path, temp_output_path, threshold)

            # Загружаем сложенное изображение
            add_img = Image.open(temp_output_path).convert("L")
            add_images_list.append(add_img)

    if not add_images_list:
        print(f"В папке {root_folder} нет изображений для сложения.")
        return

    # Складываем все объединенные изображения
    final_image = add_dark_pixels(add_images_list, threshold)

    # Сохраняем финальный результат
    final_image.save(final_output_path)
    print(f"Финальный результат сохранен в {final_output_path}")


# Пример использования
if __name__ == "__main__":
    # Папка с папками изображений
    root_folder = "./scans_jpg"

    # Пороговое значение для "близких к черному" пикселей
    threshold = 50  # Можно настроить

    # Сложение изображений в каждой папке
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            output_path = os.path.join(root_folder, f"{folder_name}_added.png")
            add_images_in_folder(folder_path, output_path, threshold)

    # Сложение всех объединенных изображений
    final_output_path = os.path.join(root_folder, "final_added.png")
    add_all_folders(root_folder, final_output_path, threshold)