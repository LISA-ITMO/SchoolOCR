import base64
import json
import re
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
from pydantic import BaseModel
import tensorflow as tf
from ultralytics import YOLO

from recognizers.code_rec import CodeRecognizer
from recognizers.table_recognizer import TableRecognizer
from recognizers.preprocess import MultiPreprocessor

class ImageProcessor:
    """Handles image loading and processing from various sources."""

    @staticmethod
    def is_pdf(file_data: bytes) -> bool:
        """Check if the input data is a PDF."""
        return len(file_data) > 4 and file_data[:4] == b'%PDF'

    @staticmethod
    def pdf_to_image(pdf_data: bytes) -> np.ndarray:
        """Convert PDF to image with 300 DPI resolution."""
        try:
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            page = doc.load_page(0)
            zoom = 300 / 72  # Convert from 72 DPI to 300 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"PDF conversion failed: {str(e)}")

    @staticmethod
    def decode_image(image_data: bytes) -> np.ndarray:
        """Decode image from bytes or PDF."""
        try:
            if ImageProcessor.is_pdf(image_data):
                return ImageProcessor.pdf_to_image(image_data)

            image_np = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid image format")
            return ImageProcessor.resize_to_target(image)
        except Exception as e:
            raise ValueError(f"Image decoding failed: {str(e)}")

    @staticmethod
    def resize_to_target(image: np.ndarray, target_width: int = 2480, target_height: int = 3505) -> np.ndarray:
        """Resize image to target dimensions."""
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def extract_region(image: np.ndarray, coords: Dict[str, int]) -> np.ndarray:
        """Extract region from image using coordinates."""
        x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
        return image[y1:y2, x1:x2]


class TextRecognizer:
    """Handles text recognition from image regions."""

    REPLACEMENTS = {
        "|": "1",
        "!": "1",
        "&": "8",
        "?": "7",
        ",": ".",
        "\n": "."
    }

    def __init__(self):
        self.preprocessor = MultiPreprocessor()
        self.whitelist = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЫЭЮЯабвгдежзиклмнопрстуфхцчшщыэюя.0123456789"

    def recognize_text(self, region_img: np.ndarray) -> str:
        """Recognize text from image region."""
        try:
            processed_img = self.preprocessor.preprocess(region_img, mode="general_ocr")
            if processed_img is None:
                return ""

            custom_config = f'-c tessedit_char_whitelist="{self.whitelist}" --psm 6'
            text = pytesseract.image_to_string(processed_img, lang='rus', config=custom_config).strip()

            for old, new in self.REPLACEMENTS.items():
                text = text.replace(old, new)

            return text
        except Exception as e:
            print(f"Text recognition error: {e}")
            return ""

    @staticmethod
    def parse_hat_text(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse subject, grade and variant from recognized text."""
        patterns = [
            # Pattern 1: Standard format
            r"^[^.]*\.\s*([^.]*)\.\s*(\d{1,2})\D*.*?(\d)\s*\.{0,2}$",
            # Pattern 2: Alternative format
            r"\.\s*([А-Яа-я ]+)\.\s*(\d{1,2})\s*[^0-9]*.*?Вариант\s*(\d+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                subject = match.group(1).lower().strip()
                grade = match.group(2).strip()
                variant = match.group(3).strip() if len(match.groups()) > 2 else None
                return subject, grade, variant

        return None, None, None


class RecognitionService:
    """Main service handling the recognition workflow."""

    def __init__(self, config_path: str = "config.json", api_keys_path: str = "api_keys.json"):
        self.config = self._load_config(config_path)
        self.api_keys = self._load_api_keys(api_keys_path)
        self.models = self._load_models()
        self.text_recognizer = TextRecognizer()
        self.image_processor = ImageProcessor()

        # Initialize recognizers
        self.code_recognizer = CodeRecognizer(debug=True)
        self.table_recognizer = TableRecognizer(debug=True)

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        """Load configuration file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    @staticmethod
    def _load_api_keys(path: str) -> set:
        """Load API keys."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return set(config.get("keys", []))
        except FileNotFoundError:
            print("API keys file not found. No API keys loaded.")
            return set()

    @staticmethod
    def _load_models() -> Dict[str, Any]:
        """Load all required ML models."""
        return {
            "mnist": tf.keras.models.load_model("./models/mnist_model.keras"),
            "extended": tf.keras.models.load_model("./models/mnist_recognation_extendend.h5"),
            "yolo": YOLO("./models/cell_detect.pt"),
            "yolo_extra": YOLO("./models/cell_detect_extra.pt")
        }

    def validate_api_key(self, api_key: str) -> bool:
        """Validate the provided API key."""
        return api_key in self.api_keys

    def recognize_image(self, image_data: bytes) -> Dict[str, Any]:
        """Main recognition pipeline."""
        errors = []
        warnings = []
        result = {
            "subject": None,
            "grade": None,
            "variant": None,
            "participant_code": None,
            "total_score": 0,
            "scores": {},
            "errors": errors,
            "warnings": warnings
        }

        try:
            # Image processing
            image = self.image_processor.decode_image(image_data)

            # Header recognition
            hat_text = self._recognize_hat(image, errors)
            subject, grade, variant = self.text_recognizer.parse_hat_text(hat_text)
            result.update({"subject": subject, "grade": grade, "variant": variant})

            # Code recognition
            result["participant_code"] = self._recognize_code(image, errors)

            print("code ok")

            # Table recognition
            task_numbers, digit_predictions = self._recognize_table(image, subject, grade, errors, warnings)

            if digit_predictions:
                self._process_digit_predictions(result, task_numbers, digit_predictions, warnings)

            # Clean empty error/warning lists
            result["errors"] = errors if errors else None
            result["warnings"] = warnings if warnings else None

            return result

        except Exception as e:
            errors.append(f"Processing error: {str(e)}")
            result["errors"] = errors
            return result

    def _recognize_hat(self, image: np.ndarray, errors: List[str]) -> str:
        """Recognize text from header region."""
        try:
            hat_region = self.image_processor.extract_region(image, self.config["regions"]["hat"])
            hat_text = self.text_recognizer.recognize_text(hat_region)

            if not hat_text:
                hat_region = self.image_processor.extract_region(image, self.config["regions"]["hat_reserve"])
                hat_text = self.text_recognizer.recognize_text(hat_region)

            return hat_text
        except Exception as e:
            errors.append("Failed to recognize header text")
            return ""

    def _recognize_code(self, image: np.ndarray, errors: List[str]) -> Optional[str]:
        """Recognize participant code."""
        try:
            code_region = self.image_processor.extract_region(image, self.config["regions"]["code"])
            return self.code_recognizer.recognize(code_region, self.models["extended"])
        except Exception:
            errors.append("Failed to recognize participant code")
            return None

    def _recognize_table(
            self,
            image: np.ndarray,
            subject: Optional[str],
            grade: Optional[str],
            errors: List[str],
            warnings: List[str]
    ) -> Tuple[Optional[List[str]], Optional[List[Tuple[int, float]]]]:
        """Recognize table content with fallback strategies."""
        config = None
        if subject and grade:
            key = f"{subject.replace(' ', '')} {grade}"
            if key in self.config:
                config = self.config[key]

        # First attempt with primary model
        try:
            result = self.table_recognizer.recognize(
                image,
                self.models["extended"],
                self.models["yolo"],
                config=config
            )

            if config:
                # In config mode, result is just digit predictions
                if result is not None:
                    task_numbers = [str(i) for i in range(1, len(result) + 1)]
                    return task_numbers, result
            else:
                # In auto mode, result is (task_numbers, digit_predictions)
                return result

        except Exception as e:
            warnings.append(f"Primary recognition failed: {str(e)}")

        # Fallback to secondary model
        try:
            result = self.table_recognizer.recognize(
                image,
                self.models["extended"],
                self.models["yolo_extra"],
                config=config
            )

            if config:
                if result is not None:
                    task_numbers = [str(i) for i in range(1, len(result) + 1)]
                    return task_numbers, result
            else:
                return result

        except Exception as e:
            errors.append("All table recognition attempts failed")
            return None, None

    def _process_digit_predictions(
            self,
            result: Dict[str, Any],
            task_numbers: List[str],
            digit_predictions: List[Tuple[int, float]],
            warnings: List[str]
    ) -> None:
        """Process digit predictions into final scores."""
        low_confidence = []

        for i, (digit, prob) in enumerate(digit_predictions):
            digit = int(digit)
            prob = round(float(prob), 2)

            if i < len(task_numbers):
                task_name = task_numbers[i]
                display_digit = '-' if digit == 10 else ('x' if digit == 11 else digit)
                result["scores"][task_name] = (display_digit, prob)

                if prob < 0.6:
                    low_confidence.append(task_name)

                if digit not in [10, 11]:
                    result["total_score"] += digit

        if low_confidence:
            warnings.append(f"Low confidence predictions for tasks: {', '.join(low_confidence)}")