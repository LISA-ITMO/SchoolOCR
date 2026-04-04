import io
import json
import time
import uuid
import asyncio
import traceback
from uuid import UUID

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image

from app.db.Db import Db
from app.db.MinioClient import MinioClient
from app.services.recognizer import DocumentRecognizer

router = APIRouter(prefix="/recognize", tags=["recognize"])

db_instance = Db()


def is_pdf(file_header: bytes) -> bool:
    return file_header.startswith(b"%PDF-")


def is_image(file_header: bytes) -> bool:
    image_signatures = [
        b"\xff\xd8\xff",
        b"\x89PNG\r\n\x1a\n",
    ]
    return any(file_header.startswith(sig) for sig in image_signatures)


def _log_future(fut, recognize_id: str):
    try:
        fut.result()
    except Exception as e:
        print(f"[recognize_job FAILED] id={recognize_id}: {e}")
        traceback.print_exc()
        try:
            Db().query(
                "UPDATE recognize_results SET completion_percent=%s WHERE id=%s",
                (0, recognize_id),
            )
        except Exception:
            pass


def recognize_job(recognize_id: str, data: bytes, header: bytes):
    db = Db()
    minio = MinioClient()
    recognizer = DocumentRecognizer(enable_logging=True)

    db.query(
        "UPDATE recognize_results SET completion_percent=%s, results=%s WHERE id=%s",
        (0, json.dumps({"items": []}), recognize_id),
    )

    if is_pdf(header):
        images = convert_from_bytes(data)
    else:
        image = Image.open(io.BytesIO(data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        images = [image]

    recognized = []

    for idx, image in enumerate(images):
        try:
            recognized_document = recognizer.recognize(image)

            try:
                object_name = f"{recognize_id}_{idx}.jpg"
                bytes_image = io.BytesIO()
                image.save(bytes_image, format="JPEG")
                minio.upload_bytes(
                    data=bytes_image.getvalue(),
                    object_name=object_name,
                    content_type="image/jpeg",
                )
                recognized_document["image_url"] = minio.get_public_url(object_name)
                recognized_document["id"] = str(uuid.uuid1())
            except Exception:
                print("Ошибка при формировании ссылки на скрин бланка")

            recognized.append(recognized_document)

        except Exception as e:
            print(e)
            traceback.print_exc()

        actual_percent = round(100 * (idx + 1) / max(1, len(images)))
        db.query(
            "UPDATE recognize_results SET completion_percent=%s, results=%s WHERE id=%s",
            (actual_percent, json.dumps({"items": recognized}), recognize_id),
        )

    db.query(
        "UPDATE recognize_results SET completion_percent=%s WHERE id=%s",
        (100, recognize_id),
    )


async def wait_until_done(recognize_id: str, timeout_s: int = 600, poll_s: float = 0.5):
    start = time.time()
    while True:
        rows = db_instance.query_get(
            "SELECT completion_percent FROM recognize_results WHERE id=%s;",
            (recognize_id,),
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Task not found")

        if int(rows[0][0] or 0) >= 100:
            return

        if time.time() - start > timeout_s:
            raise HTTPException(status_code=504, detail="Recognition timeout")

        await asyncio.sleep(poll_s)


@router.get("/create_task")
async def create_recognize_task():
    recognize_id = str(uuid.uuid4())
    db_instance.query(
        "INSERT INTO recognize_results (id, completion_percent, results) values (%s, %s, %s)",
        (recognize_id, 0, json.dumps({"items": []})),
    )
    return JSONResponse(content={"task_id": recognize_id})


@router.post("/{recognize_id}/start")
async def recognize_start(request: Request, recognize_id: UUID, file: UploadFile = File(...)):
    recognize_id_str = str(recognize_id)

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

    rows = db_instance.query_get(
        "SELECT id FROM recognize_results WHERE id=%s;", (recognize_id_str,)
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Task not found.")

    db_instance.query(
        "UPDATE recognize_results SET completion_percent=%s, results=%s WHERE id=%s",
        (0, json.dumps({"items": []}), recognize_id_str),
    )

    fut = request.app.state.executor.submit(recognize_job, recognize_id_str, data, header)
    fut.add_done_callback(lambda f: _log_future(f, recognize_id_str))

    return JSONResponse(status_code=202, content={"task_id": recognize_id_str})


@router.post("")
async def recognize_compat(request: Request, file: UploadFile = File(...)):
    recognize_id = str(uuid.uuid4())
    db_instance.query(
        "INSERT INTO recognize_results (id, completion_percent, results) values (%s, %s, %s)",
        (recognize_id, 0, json.dumps({"items": []})),
    )

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

    fut = request.app.state.executor.submit(recognize_job, recognize_id, data, header)
    fut.add_done_callback(lambda f: _log_future(f, recognize_id))

    await wait_until_done(recognize_id, timeout_s=600, poll_s=0.5)

    rows = db_instance.query_get(
        "SELECT results FROM recognize_results WHERE id=%s;", (recognize_id,)
    )
    if not rows or rows[0][0] is None:
        return JSONResponse(content={"items": []})

    results = rows[0][0]
    if isinstance(results, str):
        results = json.loads(results)

    return JSONResponse(content=results.get("items", []))


@router.get("/{id}/get_is_ready")
def get_is_ready(id: UUID):
    id_str = str(id)
    rows = db_instance.query_get(
        "SELECT completion_percent FROM recognize_results WHERE id=%s;", (id_str,)
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Task not found")

    percent = int(rows[0][0] or 0)
    return JSONResponse(content={"is_ready": percent == 100, "completion_percent": percent})


@router.get("/{id}")
def get_recognize_result(id: str):
    id_str = str(id)
    rows = db_instance.query_get(
        "SELECT completion_percent, results FROM recognize_results WHERE id=%s;", (id_str,)
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Task not found")

    completion_percent, results = rows[0]
    completion_percent = int(completion_percent or 0)

    if results is None:
        return JSONResponse(
            content={
                "is_ready": completion_percent >= 100,
                "completion_percent": completion_percent,
                "items": [],
            }
        )

    if isinstance(results, str):
        try:
            results = json.loads(results)
        except Exception:
            raise HTTPException(status_code=500, detail="Invalid results format in DB")

    return JSONResponse(
        content={
            "is_ready": completion_percent >= 100,
            "completion_percent": completion_percent,
            "items": results.get("items", []),
        }
    )
