import base64
from fastapi import HTTPException

from app.config import config
from app.ml.loader import get_extended_model
from app.services.image_utils import decode_image
from app.services.header_recognizer import try_extract_subject_grade_variant
from app.services.code_recognizer import recognize_code_from_image
from app.services.table_recognizer import recognize_scores_table


def recognize_document(request):
    errors = []
    warnings = []

    try:
        image_data = base64.b64decode(request.image_base64)
        image = decode_image(image_data)

        subject, grade, variant = try_extract_subject_grade_variant(image, config)
        if not subject or not grade:
            errors.append("Не удалось определить предмет, класс или вариант")

        code = recognize_code_from_image(image, config, get_extended_model())
        if not code:
            errors.append("Не удалось распознать код участника")

        task_dict, total_score, low_confidence = recognize_scores_table(image)
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
            "errors": errors if errors else None,
            "warnings": warnings if warnings else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
