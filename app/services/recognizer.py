from fastapi import HTTPException

from app.config import config
from app.ml.loader import get_extended_model
from app.services.image_utils import decode_image
from app.services.header_recognizer import try_extract_subject_grade_variant
from app.services.code_recognizer import recognize_code_from_image
from app.services.table_recognizer import recognize_scores_table
from io import BytesIO


def convertToJpeg(im):
    with BytesIO() as f:
        im.save(f, format='JPEG')
        return f.getvalue()
    
def recognize_document(image_png):
    errors = []
    warnings = []

    try:
        bytesJpeg = convertToJpeg(image_png)
        image = decode_image(bytesJpeg)
        print(image)

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
