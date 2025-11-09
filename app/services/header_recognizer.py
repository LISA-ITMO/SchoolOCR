import re
import pytesseract
from typing import Optional, Tuple
from app.services.image_utils import extract_region
from app.preprocessing.general import preprocess_general
from app.constants import replacements, WHITELIST


class HeaderRecognizer:
    def __init__(self, config: dict):
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self):
        self.pattern1 = re.compile(r"^[^.]*\.\s*([^.]*)\.\s*(\d{1,2})\D*.*?(\d)\s*\.{0,2}$", re.IGNORECASE)
        self.pattern2 = re.compile(r"\.\s*([А-Яа-яёЁ ]+)\.\s*(\d{1,2})\s*[^0-9]*.*?Вариант\s*(\d+)", re.IGNORECASE)

    def recognize_hat_text(self, region_img) -> str:
        processed_img = preprocess_general(region_img)
        custom_config = f'-c tessedit_char_whitelist="{WHITELIST}" --psm 6'
        text = pytesseract.image_to_string(processed_img, lang='rus', config=custom_config).strip()

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def parse_hat_text(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        match = self.pattern1.search(text)
        if match:
            return match.group(1).lower(), match.group(2), match.group(3)

        match = self.pattern2.search(text)
        if match:
            return match.group(1).lower(), match.group(2), match.group(3)

        return None, None, None

    def extract_subject_grade_variant(self, image) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        hat_region = extract_region(image, self.config["regions"]["hat"])
        text = self.recognize_hat_text(hat_region)
        subject, grade, variant = self.parse_hat_text(text)

        if not subject or not grade:
            hat_region = extract_region(image, self.config["regions"]["hat_reserve"])
            text = self.recognize_hat_text(hat_region)
            subject, grade, variant = self.parse_hat_text(text)

        return subject, grade, variant