# lambda_handler.py — Telegram webhook for Lambda (async self-invoke)
# -*- coding: utf-8 -*-
import os, json, base64, logging, requests, boto3
from app_pipeline import process_link, find_first_url, SUPPORTED

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG = f"https://api.telegram.org/bot{BOT_TOKEN}"

log = logging.getLogger()
log.setLevel(logging.INFO)

# for async self-invocation
LAMBDA_FN_NAME = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
_lambda = boto3.client("lambda")

def send_text(chat_id: int, text: str, markdown: bool = False):
    data = {"chat_id": chat_id, "text": text}
    if markdown:
        data["parse_mode"] = "Markdown"
    try:
        requests.post(f"{TG}/sendMessage", json=data, timeout=15)
    except Exception:
        log.exception("send_text failed")

def send_document(chat_id: int, filename: str, content: bytes):
    files = {"document": (filename, content, "application/json")}
    data = {"chat_id": chat_id}
    try:
        requests.post(f"{TG}/sendDocument", data=data, files=files, timeout=30)
    except Exception:
        log.exception("send_document failed")

def _process_job(chat_id: int, text: str):
    """Worker: heavy pipeline and final replies."""
    try:
        pretty, recipe_bytes = process_link(text)
        send_text(chat_id, pretty, markdown=True)
        send_document(chat_id, "recipe.json", recipe_bytes)
    except Exception as e:
        log.exception("pipeline error")
        send_text(chat_id, f"❌ שגיאה: {e}")

def lambda_handler(event, context):
    # -------- Worker branch (async self-invoke) --------
    if isinstance(event, dict) and event.get("mode") == "process":
        chat_id = event.get("chat_id")
        text = event.get("text") or ""
        if chat_id:
            _process_job(chat_id, text)
        return {"statusCode": 200, "body": "ok"}

    # -------- Webhook branch (must return 2xx fast) --------
    body_raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body_raw = base64.b64decode(body_raw).decode("utf-8")
    update = json.loads(body_raw)

    msg = update.get("message") or update.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()

    # ignore our own bot messages to prevent echo-loops
    from_user = msg.get("from") or {}
    if from_user.get("is_bot"):  # User object has is_bot flag
        return {"statusCode": 200, "body": "ok"}  # ack and drop

    if not chat_id:
        return {"statusCode": 200, "body": "ok"}

    # Commands (fast paths)
    if text.startswith("/start"):
        send_text(chat_id,
                  "שלום! שלח/י לי קישור של TikTok / Instagram / Facebook עם מתכון, "
                  "ואחזיר גרסה מסודרת + JSON.")
        return {"statusCode": 200, "body": "ok"}

    if text.startswith("/ping"):
        send_text(chat_id, "pong ✅")
        return {"statusCode": 200, "body": "ok"}

    # Must contain a supported link
    url = find_first_url(text)
    if not url or not any(root in url.lower() for root in SUPPORTED):
        send_text(chat_id, "לא זוהה קישור נתמך. שלח/י קישור של TikTok / Instagram / Facebook.")
        return {"statusCode": 200, "body": "ok"}

    # Tell user we started, then hand off to async worker and ACK immediately
    send_text(chat_id, "✅ הקישור תקין. מוריד → מתמלל → מחלץ מתכון…")

    # Fire-and-forget self-invoke so webhook returns 200 fast
    try:
        _lambda.invoke(
            FunctionName=LAMBDA_FN_NAME,
            InvocationType="Event",  # async
            Payload=json.dumps({
                "mode": "process",
                "chat_id": chat_id,
                "text": text
            }).encode("utf-8"),
        )
    except Exception:
        log.exception("async invoke failed")
        send_text(chat_id, "❌ שגיאה פנימית בהגשת המשימה. נסו שוב מאוחר יותר.")

    # Critical: return 2xx quickly so Telegram doesn't retry
    return {"statusCode": 200, "body": "ok"}
