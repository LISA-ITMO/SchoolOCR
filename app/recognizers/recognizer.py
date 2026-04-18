from fastapi import HTTPException
from io import BytesIO

from app.ml.loader import get_extended_model
from app.recognizers.image_utils import decode_image
from app.recognizers.header_recognizer import HeaderRecognizer
from app.recognizers.code_recognizer import CodeRecognizer
from app.recognizers.table_recognizer import TableRecognizer
from app.recognizers.config_manager import config_manager


class DocumentRecognizer:
    def __init__(self, debug: bool = False, enable_logging: bool = False):
        self.config_manager = config_manager
        self.model = get_extended_model()
        self.header_recognizer = HeaderRecognizer(self.config_manager)
        self.code_recognizer = CodeRecognizer(self.model, self.config_manager)
        self.table_recognizer = TableRecognizer(
            debug=debug,
            enable_logging=enable_logging,
            config_manager_instance=self.config_manager,
        )

    def _convert_to_jpeg(self, im):
        with BytesIO() as f:
            im.save(f, format="JPEG")
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
            else:
                if len(code) != 5 or not code.isdigit() or variant and code[0] != variant:
                    errors.append(f"Некорректно распознан код участника")

            task_dict, task_dict_prob_details, total_score, low_confidence = (
                self.table_recognizer.recognize_scores_table(image)
            )

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
                "warnings": warnings if warnings else None,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def recognize_document(image_png):
    recognizer = DocumentRecognizer()
    return recognizer.recognize(image_png)
