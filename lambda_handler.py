# lambda_handler.py
import os, json, base64, requests
from app_pipeline import process_link

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG = f"https://api.telegram.org/bot{BOT_TOKEN}"
# WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")  # optional

def send_text(chat_id: int, text: str, parse_mode: str | None = None):
    requests.post(f"{TG}/sendMessage", json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode})

def send_document(chat_id: int, filename: str, content: bytes):
    files = {"document": (filename, content, "application/json")}
    data = {"chat_id": chat_id}
    requests.post(f"{TG}/sendDocument", data=data, files=files)

def lambda_handler(event, context):
    # OPTIONAL: verify secret header set by Telegram if you used setWebhook ...&secret_token=...
    if WEBHOOK_SECRET:
        hdr = (event.get("headers") or {}).get("x-telegram-bot-api-secret-token") \
              or (event.get("headers") or {}).get("X-Telegram-Bot-Api-Secret-Token")
        if hdr != WEBHOOK_SECRET:
            return {"statusCode": 401, "body": "unauthorized"}

    # Parse the Telegram update JSON
    body_raw = event.get("body", "{}")
    if event.get("isBase64Encoded"):
        body_raw = base64.b64decode(body_raw).decode("utf-8")
    update = json.loads(body_raw)

    msg = update.get("message") or update.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    text = msg.get("text", "")

    if not chat_id:
        return {"statusCode": 200, "body": "ok"}  # ignore unsupported updates

    # Quick user feedback
    send_text(chat_id, "✅ Link received. Downloading → transcribing → generating recipe…")

    try:
        pretty, recipe_bytes = process_link(text)
        send_text(chat_id, pretty, parse_mode="Markdown")
        send_document(chat_id, "recipe.json", recipe_bytes)
    except Exception as e:
        send_text(chat_id, f"❌ Error: {e}")

    # NOTE: This handler returns after all work is done. For high scale or strict webhook timeouts,
    # we can switch to a 2-Lambda pattern (webhook -> SQS -> worker). We'll do that in a later stage.
    return {"statusCode": 200, "body": "ok"}
