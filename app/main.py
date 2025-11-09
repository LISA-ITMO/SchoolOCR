from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pdf2image import convert_from_bytes
from PIL import Image
from app.services.recognizer import DocumentRecognizer
from app.config import app_version

import io
import json
import base64
import asyncio

app = FastAPI(title="VPR Recognition API", version=app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_recognizer = DocumentRecognizer()


def is_pdf(file_header: bytes) -> bool:
    return file_header.startswith(b"%PDF-")


def is_image(file_header: bytes) -> bool:
    image_signatures = [
        b'\xff\xd8\xff',  # JPEG
        b'\x89PNG\r\n\x1a\n',  # PNG
    ]
    return any(file_header.startswith(sig) for sig in image_signatures)


@app.get("/")
def root():
    return "API для распознавания титульных листов ВПР работ"


@app.get("/version")
def version():
    return {"version": app_version}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    allowed_pdf_types = {"application/pdf", "application/x-pdf", "application/octet-stream"}
    allowed_image_types = {"image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff"}

    content_type_ok = (file.content_type in allowed_pdf_types or file.content_type in allowed_image_types)

    header = await file.read(12)
    await file.seek(0)
    magic_ok = is_pdf(header) or is_image(header)

    if not (content_type_ok or magic_ok):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ожидается PDF-файл или изображение (JPEG, PNG, GIF, BMP, TIFF). Проверьте формат и попробуйте снова."
        )

    data = await file.read()

    recognized = []
    try:
        if is_pdf(header):
            images = convert_from_bytes(data)
        else:
            image = Image.open(io.BytesIO(data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            images = [image]

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось прочитать файл. Убедитесь, что файл не повреждён."
        )

    for image in images:
        try:
            recognized_document = document_recognizer.recognize(image)
            recognized.append(recognized_document)
        except Exception:
            print("incorrect image")

    return JSONResponse(content=recognized)


@app.post("/recognize/stream")
async def recognize_stream(file: UploadFile = File(...)):
    allowed_pdf_types = {"application/pdf", "application/x-pdf", "application/octet-stream"}
    allowed_image_types = {"image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff"}

    content_type_ok = (file.content_type in allowed_pdf_types or file.content_type in allowed_image_types)

    header = await file.read(12)
    await file.seek(0)
    magic_ok = is_pdf(header) or is_image(header)

    if not (content_type_ok or magic_ok):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ожидается PDF-файл или изображение (JPEG, PNG, GIF, BMP, TIFF). Проверьте формат и попробуйте снова."
        )

    data = await file.read()

    try:
        if is_pdf(header):
            images = convert_from_bytes(data)
        else:
            image = Image.open(io.BytesIO(data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            images = [image]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось прочитать файл. Убедитесь, что файл не повреждён."
        )

    async def gen():
        for idx, pil_image in enumerate(images):
            payload = {"page_index": idx, "image": None, "result": None}
            try:
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                payload["image"] = f"data:image/png;base64,{b64}"

                try:
                    result = document_recognizer.recognize(pil_image)
                    payload["result"] = result
                except Exception as e:
                    payload["result"] = None
                    payload["error"] = f"Ошибка распознавания: {e}"
            except Exception as e:
                payload["error"] = f"Ошибка подготовки страницы: {e}"

            line = json.dumps(payload, ensure_ascii=False) + "\n"
            yield line.encode("utf-8")

            await asyncio.sleep(0)

    return StreamingResponse(gen(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)