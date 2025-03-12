import os
import fitz  # PyMuPDF
from PIL import Image  # Pillow


# Функция для преобразования PDF в JPG с высоким качеством
def pdf_to_images(pdf_path, output_folder, dpi=300):
    """
    Преобразует страницы PDF в изображения JPG с высоким качеством.

    :param pdf_path: Путь к PDF-файлу
    :param output_folder: Папка для сохранения изображений
    :param dpi: Разрешение (точек на дюйм)
    """
    # Открываем PDF-файл
    pdf_document = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Создаем папку для изображений
    os.makedirs(output_folder, exist_ok=True)

    # Матрица для увеличения DPI
    zoom = dpi / 72  # Коэффициент масштабирования
    matrix = fitz.Matrix(zoom, zoom)

    # Обрабатываем каждую страницу
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)

        # Рендерим страницу в изображение с высоким DPI
        pix = page.get_pixmap(matrix=matrix)

        # Сохраняем изображение
        image_path = os.path.join(output_folder, f"{pdf_name}_page_{page_num + 1}.jpg")
        pix.save(image_path)

        # Опционально: конвертируем в JPG с помощью Pillow (если нужно изменить качество)
        img = Image.open(image_path)
        img.save(image_path, "JPEG", quality=100)  # Максимальное качество

    pdf_document.close()


# Основная функция для обработки всех PDF-файлов в папке
def process_pdf_folder(input_folder, output_root_folder, dpi=300):
    """
    Обрабатывает все PDF-файлы в папке и сохраняет их страницы как JPG с высоким качеством.

    :param input_folder: Папка с PDF-файлами
    :param output_root_folder: Корневая папка для сохранения изображений
    :param dpi: Разрешение (точек на дюйм)
    """
    # Создаем корневую папку для JPG
    os.makedirs(output_root_folder, exist_ok=True)

    # Обрабатываем каждый PDF-файл
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            pdf_name = os.path.splitext(filename)[0]

            # Создаем папку для текущего PDF
            output_folder = os.path.join(output_root_folder, pdf_name)
            os.makedirs(output_folder, exist_ok=True)

            # Преобразуем PDF в JPG
            print(f"Обработка файла: {filename}")
            pdf_to_images(pdf_path, output_folder, dpi)
            print(f"Файл {filename} обработан. Изображения сохранены в {output_folder}")


# Пример использования
if __name__ == "__main__":
    input_folder = "./Сканы титульников"  # Папка с PDF-файлами
    output_root_folder = "./scans_jpg"  # Корневая папка для JPG
    dpi = 300  # Увеличенное разрешение

    process_pdf_folder(input_folder, output_root_folder, dpi)