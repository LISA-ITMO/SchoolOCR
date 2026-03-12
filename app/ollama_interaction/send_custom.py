import base64
import requests
import json


def send_image_to_llm(image_bytes, prompt, api_url, model, api_key=None):

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
        "stream": False
    }

    headers = {
        "Content-Type": "application/json"
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        api_url,
        json=payload,
        headers=headers,
        timeout=600
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    data = response.json()
    return data["message"]["content"].strip()