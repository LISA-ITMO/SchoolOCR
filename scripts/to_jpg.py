import fitz  # PyMuPDF
from PIL import Image
import os


def pdf_to_jpg(pdf_path, output_folder, dpi=300):
    pdf_document = fitz.open(pdf_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(os.path.join(output_folder, f"page_{page_number + 1}.jpg"), "JPEG", quality=100)
    pdf_document.close()


pdf_path = "Сканы титульников/ВПР русский 7 класс титул.pdf"
output_folder = f"Сканы титульников/Rus7"
pdf_to_jpg(pdf_path, output_folder, dpi=300)
