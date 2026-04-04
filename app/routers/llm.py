import io
import json

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image

from app.services.config_manager import config_manager
from app.ollama_interaction.send_custom import send_image_to_llm
from app.ollama_interaction.get_api_keys import get_llm_api_key

router = APIRouter(prefix="/llm", tags=["llm"])


def is_pdf(file_header: bytes) -> bool:
    return file_header.startswith(b"%PDF-")


def is_image(file_header: bytes) -> bool:
    image_signatures = [
        b"\xff\xd8\xff",
        b"\x89PNG\r\n\x1a\n",
    ]
    return any(file_header.startswith(sig) for sig in image_signatures)


@router.post("/recognize")
async def llm_recognize(
    file: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    model: str | None = Form(default=None),
    api_key: str | None = Form(default=None),
):
    allowed_pdf_types = {"application/pdf", "application/x-pdf", "application/octet-stream"}
    allowed_image_types = {"image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff"}

    content_type_ok = (
        file.content_type in allowed_pdf_types or file.content_type in allowed_image_types
    )
    header = await file.read(12)
    await file.seek(0)
    magic_ok = is_pdf(header) or is_image(header)

    if not (content_type_ok or magic_ok):
        raise HTTPException(status_code=400, detail="Ожидается PDF-файл или изображение")

    data = await file.read()

    llm_cfg = config_manager.get_llm_config()
    prompt = prompt or llm_cfg["prompt"]
    model = model or "qwen3-vl:235b"
    api_url = "https://ollama.com/api/chat"

    if not api_key:
        api_key = get_llm_api_key()

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Не передан API key в теле запроса и не найден в конфигурации",
        )

    if is_pdf(header):
        images = convert_from_bytes(data)
        if not images:
            raise HTTPException(status_code=400, detail="PDF не содержит страниц")
        image = images[0]
    else:
        image = Image.open(io.BytesIO(data))
        if image.mode != "RGB":
            image = image.convert("RGB")

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG", quality=95)

    try:
        result = send_image_to_llm(
            image_bytes=image_bytes.getvalue(),
            prompt=prompt,
            api_url=api_url,
            model=model,
            api_key=api_key,
        )
        try:
            return JSONResponse(content=json.loads(result))
        except Exception:
            return JSONResponse(content={"result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
