import json
import os

current_dir = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(current_dir, "config.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

app_version = os.getenv("BUILD_VERSION", "meow")
