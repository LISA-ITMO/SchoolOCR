from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pdf2image import convert_from_bytes
from app.services.recognizer import recognize_document
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

@app.get("/")
def root():
    return "API для распознавания титульных листов ВПР работ"

@app.get("/version")
def version():
    return {"version": app_version}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    allowed_types = {"application/pdf", "application/x-pdf", "application/octet-stream"}
    content_type_ok = (file.content_type in allowed_types)

    header = await file.read(5)
    await file.seek(0)
    magic_ok = header.startswith(b"%PDF-")

    if not (content_type_ok or magic_ok):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ожидается PDF-файл (application/pdf). Проверьте формат и попробуйте снова."
        )

    data = await file.read()

    recognized = []
    try:
        images = convert_from_bytes(data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось прочитать PDF. Убедитесь, что файл не повреждён."
        )

    for image in images:
        try:
            recognized_document = recognize_document(image)
            recognized.append(recognized_document)
        except Exception:
            print("incorrect image")

    return JSONResponse(content=recognized)

@app.post("/recognize/stream")
async def recognize_stream(file: UploadFile = File(...)):
    allowed_types = {"application/pdf", "application/x-pdf", "application/octet-stream"}
    content_type_ok = (file.content_type in allowed_types)

    header = await file.read(5)
    await file.seek(0)
    magic_ok = header.startswith(b"%PDF-")

    if not (content_type_ok or magic_ok):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ожидается PDF-файл (application/pdf). Проверьте формат и попробуйте снова."
        )

    data = await file.read()

    try:
        images = convert_from_bytes(data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось прочитать PDF. Убедитесь, что файл не повреждён."
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
                    result = recognize_document(pil_image)
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
