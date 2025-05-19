import os
import fitz  # PyMuPDF


def split_pdfs_in_folder(input_folder, output_base_folder=None):
    """
    Разделяет все PDF-файлы в input_folder на отдельные страницы.
    Каждый PDF сохраняется в подпапке output_base_folder (или рядом с исходным файлом, если не указано).
    """
    # Если выходная папка не указана, используем ту же папку, что и входная
    if output_base_folder is None:
        output_base_folder = input_folder
    else:
        os.makedirs(output_base_folder, exist_ok=True)

    # Проходим по всем PDF-файлам в папке
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.pdf'):
            continue  # Пропускаем не-PDF файлы

        pdf_path = os.path.join(input_folder, filename)

        # Создаем подпапку для этого PDF (без расширения .pdf)
        pdf_name_without_ext = os.path.splitext(filename)[0]
        output_folder = os.path.join(output_base_folder, pdf_name_without_ext)
        os.makedirs(output_folder, exist_ok=True)

        # Открываем PDF
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)

        # Сохраняем каждую страницу как отдельный PDF
        for page_num in range(total_pages):
            output_filename = f"page_{page_num + 1}.pdf"
            output_path = os.path.join(output_folder, output_filename)

            # Создаем новый PDF с одной страницей
            single_page_pdf = fitz.open()
            single_page_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
            single_page_pdf.save(output_path)
            single_page_pdf.close()

        pdf_document.close()
        print(f"✅ '{filename}' → сохранено {total_pages} страниц в '{output_folder}'")


if __name__ == "__main__":
    # Указываем папку с исходными PDF-файлами
    input_folder = "check_school/Nailya"  # Замените на свой путь

    # Указываем выходную папку (если None, то сохранит рядом с исходными файлами)
    output_folder = "check_school/Nailya_splitted"  # Можно оставить None

    split_pdfs_in_folder(input_folder, output_folder)