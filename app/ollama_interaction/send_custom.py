import base64
import requests
import json


def send_image_to_llm(image_bytes, prompt, api_url, model, api_key):
    if not api_key:
        raise RuntimeError("API key is required for cloud LLM requests")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }
        ],
        "stream": False,
        "format": "json"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(
        api_url,
        headers=headers,
        data=json.dumps(payload),
        timeout=600
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