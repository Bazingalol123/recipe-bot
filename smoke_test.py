import os

# Load environment variables from .venv/.env
from dotenv import load_dotenv
load_dotenv(".venv/.env")

import lambda_handler

# Minimal Telegram event with /ping command
event = {
    "body": '{"message": {"chat": {"id": 12345}, "text": "https://www.instagram.com/reel/DNkxw_MMDmH/?"}}',
    "isBase64Encoded": False
}
result = lambda_handler.lambda_handler(event, None)
print(result)