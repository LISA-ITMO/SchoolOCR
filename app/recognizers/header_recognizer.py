import re
import pytesseract
from typing import Optional, Tuple
from app.recognizers.image_utils import extract_region
from app.preprocessing.general import preprocess_general
from app.constants import replacements, WHITELIST
from app.recognizers.config_manager import ConfigManager


class HeaderRecognizer:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
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
        hat_coords = self.config_manager.get_region_coordinates("hat")
        if not hat_coords:
            return None, None, None
        hat_region = extract_region(
            image,
            {"x1": hat_coords[0], "y1": hat_coords[1], "x2": hat_coords[2], "y2": hat_coords[3]},
        )
        text = self.recognize_hat_text(hat_region)
        subject, grade, variant = self.parse_hat_text(text)

        if not subject or not grade:
            hat_reserve_coords = self.config_manager.get_region_coordinates("hat_reserve")
            if not hat_reserve_coords:
                return subject, grade, variant
            hat_region = extract_region(
                image,
                {
                    "x1": hat_reserve_coords[0],
                    "y1": hat_reserve_coords[1],
                    "x2": hat_reserve_coords[2],
                    "y2": hat_reserve_coords[3],
                },
            )
            text = self.recognize_hat_text(hat_region)
            subject, grade, variant = self.parse_hat_text(text)

        return subject, grade, variant
