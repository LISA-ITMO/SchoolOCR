import os
import sys
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import io
from datetime import datetime

# Добавляем путь для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.detection.yolo_cells import extract_table_rows
from app.ml.loader import get_yolo_model, get_yolo_model_extra


class TableCellsExtractor:
    def __init__(self, save_cells: bool = False, save_dir: str = None):
        self.yolo = get_yolo_model()
        self.yolo_extra = get_yolo_model_extra()
        self.should_save_cells = save_cells
        self.save_dir = save_dir or self._get_default_save_dir()

        if self.should_save_cells:
            self._create_save_directory()

    def _get_default_save_dir(self) -> str:
        """Создает папку для сохранения ячеек с timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join("extracted_cells", f"cells_{timestamp}")

    def _create_save_directory(self):
        """Создает папку для сохранения ячеек"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _get_cell_width(self, cell):
        return cell[2] - cell[0]

    def _filter_cells(self, table_rows):
        """Фильтрует ячейки таблицы"""
        if not table_rows:
            return None, None

        if len(table_rows) % 2 != 0:
            table_rows = [row for row in table_rows if len(row) > 3]
            if len(table_rows) % 2 != 0:
                return None, None

        if len(table_rows) == 2:
            return table_rows[0][1:-2], table_rows[1][1:-2]

        elif len(table_rows) == 4:
            first_cell_width = self._get_cell_width(table_rows[2][0])
            second_cell_width = self._get_cell_width(table_rows[2][1])

            if first_cell_width - second_cell_width > 30:
                return table_rows[0][1:] + table_rows[2][1:-2], table_rows[1][1:] + table_rows[3][1:-2]
            else:
                return table_rows[0][1:] + table_rows[2][:-2], table_rows[1][1:] + table_rows[3][:-2]

        elif len(table_rows) == 6:
            return table_rows[1][1:] + table_rows[4][1:-2], table_rows[2][1:] + table_rows[5][1:-2]

        return None, None

    def extract_cells_from_image(self, image: np.ndarray, source_name: str = "unknown"):
        """
        Извлекает ячейки из изображения без распознавания
        Returns: List of (filename, cell_image)
        """
        # Пробуем основную модель YOLO
        table_rows = extract_table_rows(image, self.yolo)
        filtered_cells_tasks, filtered_cells_mnist = self._filter_cells(table_rows)

        # Если не нашли, пробуем дополнительную модель
        if not filtered_cells_mnist:
            table_rows = extract_table_rows(image, self.yolo_extra)
            filtered_cells_tasks, filtered_cells_mnist = self._filter_cells(table_rows)

        if not filtered_cells_mnist:
            return []

        # Обрабатываем дубликаты
        if len(filtered_cells_mnist) != len(filtered_cells_tasks):
            i = 0
            while i < len(filtered_cells_mnist) - 1:
                current_x = filtered_cells_mnist[i][0]
                next_x = filtered_cells_mnist[i + 1][0]
                if abs(next_x - current_x) <= 50:
                    filtered_cells_mnist.pop(i + 1)
                else:
                    i += 1

        if not filtered_cells_mnist:
            return []

        # Извлекаем изображения ячеек
        extracted_cells = []
        for i, cell in enumerate(filtered_cells_mnist):
            x1, y1, x2, y2 = map(int, cell)
            cell_img = image[y1:y2, x1:x2]

            if cell_img.size == 0:
                continue

            # Генерируем имя файла
            filename = f"{source_name}_cell_{i + 1:03d}.png"
            extracted_cells.append((filename, cell_img))

        return extracted_cells

    def save_extracted_cells(self, cells):
        """Сохраняет извлеченные ячейки"""
        if not self.should_save_cells or not cells:
            return

        for filename, cell_img in cells:
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, cell_img)

        print(f"Сохранено {len(cells)} ячеек в {self.save_dir}")


def pdf_to_image(pdf_path):
    """Конвертирует PDF в изображение с DPI 300"""
    try:
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()

        doc = fitz.open(stream=pdf_data, filetype="pdf")
        images = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Устанавливаем DPI 300
            zoom = 300 / 72  # 72 - стандартный DPI в PyMuPDF
            mat = fitz.Matrix(zoom, zoom)

            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)

        doc.close()
        return images

    except Exception as e:
        print(f"Ошибка при конвертации PDF {pdf_path}: {e}")
        return []


def find_pdfs_recursive(start_dir):
    """Рекурсивно находит все PDF файлы"""
    pdf_files = []

    for root, dirs, files in os.walk(start_dir):
        if 'extracted_cells' in root:
            continue

        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)

    return pdf_files


def main():
    # ХАРДКОД ПУТЕЙ
    input_dir = "../check_school/Nailya_splitted"
    output_dir = "./extracted_cells"

    if not os.path.exists(input_dir):
        print(f"Ошибка: входная папка {input_dir} не существует")
        return

    # Создаем экстрактор
    print("Инициализация TableCellsExtractor...")
    extractor = TableCellsExtractor(save_cells=True, save_dir=output_dir)
    print("TableCellsExtractor инициализирован")

    # Находим PDF файлы
    print(f"Поиск PDF файлов в {input_dir}...")
    pdf_files = find_pdfs_recursive(input_dir)

    if not pdf_files:
        print(f"PDF файлы не найдены")
        return

    print(f"Найдено PDF файлов: {len(pdf_files)}")

    total_cells = 0

    # Обрабатываем каждый PDF
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Обрабатываем {os.path.basename(pdf_path)}")

        # Конвертируем PDF в изображения с DPI 300
        images = pdf_to_image(pdf_path)

        if not images:
            continue

        pdf_cells_count = 0

        # Обрабатываем каждую страницу
        for page_num, image in enumerate(images, 1):
            # Создаем имя источника
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            source_name = f"{pdf_name}_p{page_num}"

            # Извлекаем ячейки
            cells = extractor.extract_cells_from_image(image, source_name)

            if cells:
                # Сохраняем ячейки
                extractor.save_extracted_cells(cells)
                pdf_cells_count += len(cells)
                print(f"  Страница {page_num}: извлечено {len(cells)} ячеек")
            else:
                print(f"  Страница {page_num}: ячейки не найдены")

        total_cells += pdf_cells_count
        print(f"  Всего извлечено из PDF: {pdf_cells_count} ячеек")

    print(f"\n{'=' * 50}")
    print("Извлечение завершено!")
    print(f"Входная папка: {input_dir}")
    print(f"Выходная папка: {output_dir}")
    print(f"Обработано PDF файлов: {len(pdf_files)}")
    print(f"Всего извлечено ячеек: {total_cells}")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()