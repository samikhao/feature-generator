import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
API_BASE = os.getenv("API_BASE", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-3-235b-a22b-instruct-2507")
