import re
import pytesseract
from app.services.image_utils import extract_region
from app.preprocessing.general import preprocess_general
from app.constants import replacements, WHITELIST

def recognize_hat_text(region_img):
    processed_img = preprocess_general(region_img)
    custom_config = f'-c tessedit_char_whitelist="{WHITELIST}" --psm 6'
    text = pytesseract.image_to_string(processed_img, lang='rus', config=custom_config).strip()
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def parse_hat_text(text):
    pattern1 = re.compile(r"^[^.]*\.\s*([^.]*)\.\s*(\d{1,2})\D*.*?(\d)\s*\.{0,2}$", re.IGNORECASE)
    match = pattern1.search(text)
    if match:
        return match.group(1).lower(), match.group(2), match.group(3)

    pattern2 = re.compile(r"\.\s*([А-Яа-яёЁ ]+)\.\s*(\d{1,2})\s*[^0-9]*.*?Вариант\s*(\d+)", re.IGNORECASE)
    match = pattern2.search(text)
    if match:
        return match.group(1).lower(), match.group(2), match.group(3)

    return None, None, None

def try_extract_subject_grade_variant(image, config):
    hat_region = extract_region(image, config["regions"]["hat"])
    text = recognize_hat_text(hat_region)
    subject, grade, variant = parse_hat_text(text)

    if not subject or not grade:
        hat_region = extract_region(image, config["regions"]["hat_reserve"])
        text = recognize_hat_text(hat_region)
        subject, grade, variant = parse_hat_text(text)

    return subject, grade, variant
