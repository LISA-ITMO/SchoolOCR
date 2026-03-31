import base64
import json
from io import BytesIO

import requests
from PIL import Image, ImageOps


def _prepare_image_for_llm(
    image_bytes: bytes,
    max_side: int = 1400,
    jpeg_quality: int = 80,
) -> bytes:
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    longest_side = max(width, height)

    if longest_side > max_side:
        scale = max_side / longest_side
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        image = image.resize((new_width, new_height), Image.LANCZOS)

    output = BytesIO()
    image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    return output.getvalue()


def send_image_to_llm(
    image_bytes,
    prompt,
    api_url,
    model,
    api_key,
    max_side: int = 1400,
    jpeg_quality: int = 80,
):
    if not api_key:
        raise RuntimeError("API key is required for cloud LLM requests")

    prepared_image = _prepare_image_for_llm(
        image_bytes=image_bytes,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
    )

    image_b64 = base64.b64encode(prepared_image).decode("utf-8")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "format": "json",
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(
        api_url,
        headers=headers,
        data=json.dumps(payload),
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM request failed: {response.status_code} {response.text}"
        )

    try:
        data = response.json()
    except Exception:
        raise RuntimeError(f"Invalid JSON response: {response.text}")

    if "message" not in data or "content" not in data["message"]:
        raise RuntimeError(f"Unexpected response format: {data}")

    return data["message"]["content"].strip()