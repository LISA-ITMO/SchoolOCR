from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.recognizer import recognize_document
from app.config import app_version
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes

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

if __name__ == "__app__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
