from app.db.Db import Db
from app.db.MinioClient import MinioClient
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image
from app.services.recognizer import DocumentRecognizer
from app.config import app_version
import traceback
import io
import json
import uuid

app = FastAPI(title="VPR Recognition API", version=app_version)

db_instance = Db()
minio_client = MinioClient()


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
        b"\xff\xd8\xff",  # JPEG
        b"\x89PNG\r\n\x1a\n",  # PNG
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
    recognize_id = str(uuid.uuid1())

    db_instance.query(
        """
        INSERT INTO recognize_results (id, completion_percent)
        values (%s, %s)
        """,
        (recognize_id, 0),
    )

    allowed_pdf_types = {
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    }
    allowed_image_types = {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/tiff",
    }

    content_type_ok = (
        file.content_type in allowed_pdf_types
        or file.content_type in allowed_image_types
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

    recognized = []
    try:
        if is_pdf(header):
            images = convert_from_bytes(data)
        else:
            image = Image.open(io.BytesIO(data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            images = [image]

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось прочитать файл. Убедитесь, что файл не повреждён.",
        )

    for idx, image in enumerate(images):
        try:
            recognized_document = document_recognizer.recognize(image)

            try:
                object_name = f"{recognize_id}_{idx}.jpg"
                bytes_image = io.BytesIO()
                image.save(bytes_image, format="JPEG")
                bytes_image = bytes_image.getvalue()
                minio_client.upload_bytes(
                    data=bytes_image, object_name=object_name, content_type="image/jpeg"
                )
                recognized_document["image_url"] = minio_client.get_public_url(
                    object_name
                )
            except Exception as e:
                print(f"Ошибка при формировании ссылки на скрин бланка")

            recognized.append(recognized_document)
        except Exception as e:
            print(e)
            traceback.print_exc()
        actual_percent = round(100 * (idx + 1) / len(images))
        db_payload = (
            actual_percent,
            json.dumps({"items": [recognized]}),
            recognize_id,
        )

        db_instance.query(
            """
            UPDATE recognize_results SET
            completion_percent=%s,
            results=%s
            WHERE id=%s
            """,
            db_payload,
        )

    return JSONResponse(content=recognized)


@app.get("/recognize/{id}/get_is_ready")
def get_is_ready(id: str):
    # todo вынести запросы в репозитории, добавить валидации
    percent = db_instance.query_get(
        """
        SELECT completion_percent from recognize_results
        WHERE id=%s;
        """,
        (id,),
    )[0][0]
    is_ready = percent == 100

    return JSONResponse(content={"is_ready": is_ready, "completion_percent": percent})


@app.get("/recognize/{id}")
def get_recognize_result(id: str):
    result = db_instance.query_get(
        """
        SELECT results from recognize_results
        WHERE id=%s;
        """,
        (id,),
    )[0]

    payload = {"items": result[0]["items"][0]}
    return JSONResponse(content=payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
