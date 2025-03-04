import os
import cv2

# ВПР Математика 6 класс
hat_coords = [(284, 113), (1017, 244)]
code_coords = [(1532, 100), (2324, 264)]
table_coords = [(235, 2754), (2393, 3105)]

def crop_images_in_folder(input_folder, output_folder, x1, y1, x2, y2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image = cv2.imread(file_path)
            if image is None:
                print(f"Не удалось загрузить изображение {filename}.")
                continue

            if (x2 > image.shape[1]) or (y2 > image.shape[0]):
                print(f"Прямоугольник выходит за границы изображения {filename}.")
                continue

            cropped_image = image[y1:y2, x1:x2]
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_image)
            print(f"Изображение {filename} обрезано и сохранено в {output_path}.")


if __name__ == "__main__":
    input_folder = "output_images"
    output_folder = "cropped_tables"
    x1, y1 = 235, 2754
    x2, y2 = 2393, 3105
    crop_images_in_folder(input_folder, output_folder, x1, y1, x2, y2)