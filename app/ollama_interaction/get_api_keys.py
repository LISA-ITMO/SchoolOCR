import json
import os

API_KEYS_PATH = "config/api_keys.json"


def get_llm_api_key():

    if not os.path.exists(API_KEYS_PATH):
        return None

    with open(API_KEYS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("api_key_llm")