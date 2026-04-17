import io
import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image

from app.recognizers.recognizer import DocumentRecognizer

router = APIRouter(prefix="/recognize", tags=["recognize"])

_recognizer = None


def get_recognizer() -> DocumentRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = DocumentRecognizer()
    return _recognizer


def is_pdf(file_header: bytes) -> bool:
    return file_header.startswith(b"%PDF-")


def is_image(file_header: bytes) -> bool:
    image_signatures = [
        b"\xff\xd8\xff",
        b"\x89PNG\r\n\x1a\n",
    ]
    return any(file_header.startswith(sig) for sig in image_signatures)


@router.post("")
async def recognize_endpoint(file: UploadFile = File(...)):
    allowed_pdf_types = {"application/pdf", "application/x-pdf", "application/octet-stream"}
    allowed_image_types = {"image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff"}

    content_type_ok = (
        file.content_type in allowed_pdf_types or file.content_type in allowed_image_types
    )
    header = await file.read(12)
    await file.seek(0)
    magic_ok = is_pdf(header) or is_image(header)

    if not (content_type_ok or magic_ok):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ожидается PDF-файл или изображение (JPEG, PNG, GIF, BMP, TIFF). Проверьте формат и попробуйте снова.",
        )

    data = await file.read()

    if is_pdf(header):
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(data)
    else:
        image = Image.open(io.BytesIO(data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        images = [image]

    recognizer = get_recognizer()
    recognized = []

    for image in images:
        try:
            result = recognizer.recognize(image)
            result["id"] = str(uuid.uuid1())
            recognized.append(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=recognized)
