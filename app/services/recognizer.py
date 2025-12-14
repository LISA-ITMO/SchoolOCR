from fastapi import HTTPException
from io import BytesIO

from app.config import config
from app.ml.loader import get_extended_model
from app.services.image_utils import decode_image
from app.services.header_recognizer import HeaderRecognizer
from app.services.code_recognizer import CodeRecognizer
from app.services.table_recognizer import TableRecognizer
from app.services.config_manager import config_manager


class DocumentRecognizer:
    def __init__(self):
        self.config = config
        self.model = get_extended_model()
        self.header_recognizer = HeaderRecognizer(self.config)
        self.code_recognizer = CodeRecognizer(self.model, self.config)

    def _convert_to_jpeg(self, im):
        with BytesIO() as f:
            im.save(f, format='JPEG')
            return f.getvalue()

    def recognize(self, image_png):
        errors = []
        warnings = []

        try:
            bytes_jpeg = self._convert_to_jpeg(image_png)
            image = decode_image(bytes_jpeg)

            subject, grade, variant = self.header_recognizer.extract_subject_grade_variant(image)
            if not subject or not grade:
                errors.append("Не удалось определить предмет, класс или вариант")

            code = self.code_recognizer.recognize(image)
            if not code:
                errors.append("Не удалось распознать код участника")

            recog_cfg = config_manager.get_recognition_config()
            table_recognizer = TableRecognizer(
                use_llm=bool(recog_cfg["use_llm"]),
                confidence_threshold=float(recog_cfg["confidence_threshold"]),
                llm_trigger_threshold=float(recog_cfg["llm_trigger_threshold"]),
            )

            task_dict, task_dict_prob_details, total_score, low_confidence = table_recognizer.recognize_scores_table(image)

            if task_dict is None:
                errors.append("Не удалось распознать таблицу")
            elif low_confidence:
                warnings.append(f"Низкая уверенность в заданиях: {', '.join(low_confidence)}")

            return {
                "subject": subject,
                "grade": grade,
                "variant": variant,
                "participant_code": code,
                "total_score": total_score,
                "scores": task_dict,
                "scores_details": task_dict_prob_details,
                "errors": errors if errors else None,
                "warnings": warnings if warnings else None
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def recognize_document(image_png):
    recognizer = DocumentRecognizer()
    return recognizer.recognize(image_png)