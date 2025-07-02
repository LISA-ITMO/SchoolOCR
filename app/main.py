from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.image_request import ImageRequest
from app.services.recognizer import recognize_document
from app.config import app_version

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
def recognize(request: ImageRequest):
    try:
        return recognize_document(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__app__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
