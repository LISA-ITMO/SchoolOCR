from app.ollama_interaction.recognize_digit import classify_image_api  # импорт твоей функции

IMAGE_PATH = "extracted_cells/page_5_p1_cell_004.png"   # путь к картинке

if __name__ == "__main__":
    # читаем файл байтами
    with open(IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    result = classify_image_api(img_bytes)
    print("Распознанный символ:", result)
