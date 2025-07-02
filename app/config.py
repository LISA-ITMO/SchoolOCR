import json
import os

CONFIG_PATH = "app/config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

app_version = os.getenv("BUILD_VERSION", "meow")
