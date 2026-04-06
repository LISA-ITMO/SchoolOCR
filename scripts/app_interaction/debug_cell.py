import cv2
from app.recognizers.cell_recognizer import CellRecognizer

# --------- НАСТРОЙКА ----------
IMAGE_PATH = "organized_cells/6/page_1_p1_cell_001_prob0.985.png"
# ------------------------------


def main():
    # включаем встроенный debug
    recognizer = CellRecognizer(debug=True, enable_logging=False)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Не удалось загрузить изображение")
        return

    # прогон
    result = recognizer.recognize_cell(img, IMAGE_PATH)

    print("\n=== RESULT ===")
    print("Predicted:", result.digit)
    print("Confidence:", result.prob)
    print("All probabilities:", result.prob_all)

    # ВОТ ГЛАВНОЕ
    recognizer.show_debug_plot()


if __name__ == "__main__":
    main()