from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import base64

from services.recognition_service import RecognitionService

app = FastAPI()
service = RecognitionService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image_base64: str


@app.post("/recognize")
async def recognize_image(
        request: ImageRequest,
        authorization: str = Header(None)
) -> Dict[str, Any]:
    """API endpoint for image recognition."""
    if not service.validate_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        image_data = base64.b64decode(request.image_base64)
        return service.recognize_image(image_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)