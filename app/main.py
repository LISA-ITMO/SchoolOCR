import json
import os
import multiprocessing as mp
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from app.config import app_version
from app.routers import recognize, llm


mp.set_start_method("spawn", force=True)

_api_keys_path = os.path.join(os.path.dirname(__file__), "api_keys.json")
with open(_api_keys_path) as _f:
    _VALID_API_KEYS: set = set(json.load(_f)["keys"])

_api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(key: str = Security(_api_key_header)):
    if key not in _VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.executor = ProcessPoolExecutor(max_workers=6)
    try:
        yield
    finally:
        ex = getattr(app.state, "executor", None)
        if ex:
            ex.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="VPR Recognition API",
    version=app_version,
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recognize.router)
app.include_router(llm.router)


@app.get("/")
def root():
    return "API для распознавания титульных листов ВПР работ"


@app.get("/version")
def version():
    return {"version": app_version}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
