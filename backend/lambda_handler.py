# lambda_handler.py — Telegram webhook for Lambda
import os, json, base64, logging, requests
from app_pipeline import process_link, find_first_url, SUPPORTED

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG = f"https://api.telegram.org/bot{BOT_TOKEN}"
# WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")  # optional

log = logging.getLogger()
log.setLevel(logging.INFO)

def send_text(chat_id: int, text: str, markdown: bool = False):
    data = {"chat_id": chat_id, "text": text}
    if markdown:
        data["parse_mode"] = "Markdown"
    requests.post(f"{TG}/sendMessage", json=data, timeout=15)

def send_document(chat_id: int, filename: str, content: bytes):
    files = {"document": (filename, content, "application/json")}
    data = {"chat_id": chat_id}
    requests.post(f"{TG}/sendDocument", data=data, files=files, timeout=30)

def lambda_handler(event, context):
    # Optional secret verification (if you set setWebhook&secret_token=...)
    # if WEBHOOK_SECRET:
    #     hdrs = event.get("headers") or {}
    #     recv = hdrs.get("x-telegram-bot-api-secret-token") or hdrs.get("X-Telegram-Bot-Api-Secret-Token")
    #     if recv != WEBHOOK_SECRET:
    #         return {"statusCode": 401, "body": "unauthorized"}

    body_raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body_raw = base64.b64decode(body_raw).decode("utf-8")
    update = json.loads(body_raw)

    msg = update.get("message") or update.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()

    if not chat_id:
        return {"statusCode": 200, "body": "ok"}

    # Commands
    if text.startswith("/start"):
        send_text(chat_id,
            "שלום!  .🙏🏻שלח/י לי קישור של TikTok / Instagram / Facebook עם מתכון, "
            "ואחזיר לך גרסה מסודרת + JSON.")
        return {"statusCode": 200, "body": "ok"}

    if text.startswith("/ping"):
        send_text(chat_id, "pong ✅")
        return {"statusCode": 200, "body": "ok"}

    # Must contain a supported link
    url = find_first_url(text)
    if not url or not any(root in url.lower() for root in SUPPORTED):
        send_text(chat_id, "לא זוהה קישור נתמך. שלח/י קישור של TikTok / Instagram / Facebook.")
        return {"statusCode": 200, "body": "ok"}

    send_text(chat_id, "✅ הקישור תקין. מוריד → מתמלל → מחלץ מתכון…")
    try:
        pretty, recipe_bytes = process_link(text)
        send_text(chat_id, pretty, markdown=True)
        send_document(chat_id, "recipe.json", recipe_bytes)
    except Exception as e:
        log.exception("pipeline error")
        send_text(chat_id, f"❌ שגיאה: {e}")

    return {"statusCode": 200, "body": "ok"}