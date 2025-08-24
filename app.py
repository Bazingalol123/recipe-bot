# -*- coding: utf-8 -*-
# Telegram Recipe Extractor - Option A (with generic post-validator)
# Requirements: python-telegram-bot v20+, openai>=1.0, yt-dlp, ffmpeg, python-dotenv

from dotenv import load_dotenv
import os, re, json, logging, tempfile, subprocess, shlex, asyncio
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import subprocess
from openai import OpenAI  # LLM + Whisper

# ===== Config =====
LLM_MODEL = "gpt-4o-mini"
ASR_MODEL = "gpt-4o-mini-transcribe"
IG_MEDIA_PATH = re.compile(r"^/(reel|reels|p|tv)/([A-Za-z0-9_\-]+)/?")


# --- Logging ---
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Secrets ---
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")
if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_KEY)

# ================= Utils =================
def safe_load_json(maybe_json: str) -> dict:
    import json as _json
    s = (maybe_json or "").strip()

    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    try:
        return _json.loads(s)
    except Exception as e:
        return {"_parse_error": str(e), "_raw_preview": s[:1000]}





# --- URL helpers ---
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
SUPPORTED_DOMAINS = {"tiktok.com","instagram.com","facebook.com","fb.watch"}

def find_urls(text: str) -> list[str]:
    return URL_RE.findall(text or "")

def is_supported(url: str) -> tuple[bool, str]:
    try:
        p = urlparse(url)
        if p.scheme not in ("http","https"):
            return False, "URL must start with http(s)"
        if not p.netloc:
            return False, "URL missing domain"
        host = p.netloc.lower()
        if any(root in host for root in SUPPORTED_DOMAINS):
            return True, "Supported platform"
        return False, f"Unsupported platform: {host}"
    except Exception as e:
        return False, f"Could not parse URL ({e})"

def canonicalize_url(u: str) -> str:
    """
    Normalize social links so yt-dlp is happier.
    - Instagram: keep only scheme + host + '/{kind}/{id}/'
    - TikTok: drop query/fragment, keep path
    - Facebook: drop utm* tracking; keep watch?v=… if present
    """
    logger.info("Normalizing link and removing user parameter")
    p = urlparse(u)
    host = (p.netloc or "").lower()
    scheme = p.scheme or "https"

    if "instagram.com" in host:
        m = IG_MEDIA_PATH.match(p.path)
        if m:
            kind, media_id = m.groups()
            # Force www (IG is picky sometimes), drop query/fragment
            logger.info(("new url is:","www.instagram.com", f"/{kind}/{media_id}/" ) )
            return urlunparse((scheme, "www.instagram.com", f"/{kind}/{media_id}/", "", "", ""))
        # If path didn’t match, at least drop query/fragment
        return urlunparse((scheme, "www.instagram.com", p.path, "", "", ""))

    if "tiktok.com" in host:
        # Keep path, drop query/fragment, ensure trailing slash for consistency
        path = p.path.rstrip("/") or "/"
        return urlunparse((scheme, "www.tiktok.com", path + ("/" if not path.endswith("/") else ""), "", "", ""))

    if "facebook.com" in host or "fb.watch" in host:
        # Drop tracking params but keep meaningful ones (e.g., v in watch?v=…)
        q = dict(parse_qsl(p.query))
        for k in list(q.keys()):
            if k.startswith("utm_") or k in {"mibextid", "s", "si"}:
                q.pop(k, None)
        qstr = urlencode(q) if q else ""
        return urlunparse((scheme, host, p.path, "", qstr, ""))

    return u



# --- Shell helpers ---
def run_cmd(cmd: str) -> str:
    """
    Run a shell command, raising on failure, and return stdout text (if any).
    """
    logger.debug(f"RUN: {cmd}")
    completed = subprocess.run(
        cmd, shell=True, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if completed.stderr.strip():
        logger.debug(f"STDERR: {completed.stderr.strip()[:500]}")
    return completed.stdout

def download_media(url: str, workdir: Path) -> dict:
    """
    Fetch best video, extract mono 16 kHz WAV for ASR, and collect metadata via yt-dlp -J.
    """
    logger.info("Initializing Downlad Media function...")
    out_tmpl = str(workdir / "vid.%(ext)s")
    run_cmd(f'yt-dlp -o "{out_tmpl}" --merge-output-format mp4 {shlex.quote(url)}')

    # Pull metadata from yt-dlp -J
    meta_raw = run_cmd(f'yt-dlp -J {shlex.quote(url)}')
    try:
        meta = json.loads(meta_raw)
    except Exception:
        meta = {}
    title = meta.get("title") or ""
    desc  = meta.get("description") or ""
    tags  = meta.get("tags") or []

    video_path = None
    for f in workdir.iterdir():
        if f.name.startswith("vid.") and f.suffix.lower() in (".mp4",".mkv",".webm",".mov"):
            video_path = f
    if not video_path:
        raise RuntimeError("Could not find downloaded video file.")

    audio_path = workdir / "audio.wav"
    run_cmd(f'ffmpeg -y -i "{video_path}" -vn -ac 1 -ar 16000 "{audio_path}"')
    logger.info("Downloaded successfully")


    return {

        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "source_url": url,
        "meta": {"title": title, "description": desc, "tags": tags},
    }

# --- ASR ---
def transcribe_audio(audio_path: str) -> str:
    """
    Try Hebrew first; if it fails, retry without fixed language (auto-detect).
    """
    logger.info("Transcribing started.")
    with open(audio_path, "rb") as f:
        try:
            resp = client.audio.transcriptions.create(
                model=ASR_MODEL,
                file=f,
                language="he",
            )
            logger.info("Transcribing Finished.")
            return resp.text
        except Exception as e:
            logger.warning(f"ASR he=he failed ({e}); retrying with auto language…")
            f.seek(0)
            resp2 = client.audio.transcriptions.create(
                model=ASR_MODEL,
                file=f,
            )
            logger.info("Transcribing was not successful. Trying again.")

            return resp2.text

# ================= Schema =================
RECIPE_TOOL_PARAMETERS = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "title": {"type": "string"},               # Hebrew only
    "source_url": {"type": "string"},
    "lang": {"type": "string", "enum": ["he"]},
    "servings": {"type": ["integer", "null"]},
    "total_time_minutes": {"type": ["integer", "null"]},

    "ingredients": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "item": {"type": "string"},          # Hebrew only
          "quantity": {"type": ["number", "null"]},
          "unit": {
            "type": ["string", "null"],
            "enum": ["גרם", "מ״ל", "כפית", "כף", "כוס", "יחידה"]
          },
          "notes": {"type": ["string", "null"]}# Hebrew only
        },
        "required": ["item"]
      }
    },

    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "number": {"type": "integer"},
          "instruction": {"type": "string"},   # Hebrew only
          "time_minutes": {"type": ["integer", "null"]}
        },
        "required": ["number", "instruction"]
      }
    },

    "equipment": {"type": "array", "items": {"type": "string"}},
    "notes": {"type": "array", "items": {"type": "string"}},

    "nutrition_estimate": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "kcal_per_serving": {"type": ["number", "null"]}
      }
    },

    "confidence": {"type": "number"},

    "raw": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "transcript": {"type": "string"}
      },
      "required": ["transcript"]
    },

    # ---- Evidence fields ----
    "dish_canonical": {"type": ["string", "null"]},  # e.g., "בולונז"
    "title_source": {"type": ["string", "null"], "enum": ["transcript","metadata","both"]},
    "evidence": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "title_terms": {"type":"array","items":{"type":"string"}},
        "ingredient_spans": {"type":"array","items":{
            "type":"object",
            "properties": {"item":{"type":"string"}, "snippet":{"type":"string"}},
            "required":["item","snippet"],
            "additionalProperties": False
        }}
      }
    }
  },
  "required": ["title", "ingredients", "steps", "confidence", "raw", "lang"]
}

def parse_tool_args(args_str: str) -> dict:
    try:
        return json.loads(args_str)
    except Exception:
        pass
    return safe_load_json(args_str)

# ================= LLM: transcript/meta → JSON =================
SYSTEM_PROMPT = (
  "אתה מנוע חילוץ קולינרי. ענה בעברית בלבד.\n"
  "השתמש אך ורק בתמלול וב-metadata שסופקו (כותרת/תיאור/תגיות). "
  "אסור להמציא מרכיבים או שלבים שאינם מופיעים במפורש. אם חסר מידע – השאר null/ריק והורד confidence. "
  "נרמל יחידות ל: גרם, מ״ל, כפית, כף, כוס, יחידה. הגדר lang='he'. אין לכלול אותיות לטיניות בפלט.\n"
  "כותרת: הימנע מכותרות כלליות. אם בתמלול או ב-metadata מופיעים שמות מנה (למשל 'בולונז' / 'bolognese' / 'ragù'), "
  "עדכן title בהתאם, הגדר dish_canonical לשם המנה, ו-title_source='transcript'/'metadata'/'both'. "
  "החזר evidence: לכל מרכיב ציין snippet מהתמלול/metadata שמצדיק אותו; וכן רשימת מילות מפתח שהשפיעו על הכותרת."
)

def make_recipe(transcript: str, source_url: str, meta: dict) -> dict:
    """
    Function-calling with robust parsing and a one-time repair retry.
    """
    logger.info("Calling Open AI to build recipe")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({
            "source_url": source_url,
            "metadata": meta,
            "transcript": transcript
        }, ensure_ascii=False)}
    ]

    tools = [{
        "type": "function",
        "function": {
            "name": "submit_recipe",
            "description": "Return a structured recipe JSON.",
            "parameters": RECIPE_TOOL_PARAMETERS
        }
    }]

    def _call(messages_):
        return client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            messages=messages_,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "submit_recipe"}},
            max_tokens=1200,
            parallel_tool_calls=False
        )

    # First attempt
    resp = _call(messages)
    choice = resp.choices[0].message

    if getattr(choice, "tool_calls", None):
        args_str = choice.tool_calls[0].function.arguments
        data = parse_tool_args(args_str)
        if "_parse_error" not in data:
            try:
                data.setdefault("raw", {})["transcript"] = transcript[:1200]
                data["source_url"] = source_url
                data["lang"] = "he"
            except Exception:
                pass
            return data

    # Repair attempt: ask the model to re-emit a valid function call (no prose)
    repair_messages = [
        {"role": "system", "content":
         "הארגומנטים לכלי לא היו JSON תקין. "
         "החזר אך ורק קריאה לפונקציה submit_recipe עם JSON חוקי התואם לסכימה. ללא טקסט חופשי. "
         "שמור raw.transcript ≤ 1200 תווים."
        },
        {"role": "user", "content": json.dumps({
            "source_url": source_url,
            "metadata": meta,
            "transcript": transcript[:3500]
        }, ensure_ascii=False)}
    ]
    resp2 = _call(repair_messages)
    choice2 = resp2.choices[0].message
    if getattr(choice2, "tool_calls", None):
        args_str2 = choice2.tool_calls[0].function.arguments
        data2 = parse_tool_args(args_str2)
        try:
            data2.setdefault("raw", {})["transcript"] = transcript[:1200]
            data2["source_url"] = source_url
            data2["lang"] = "he"
        except Exception:
            pass
        return data2

    return {"_parse_error": "No tool call returned", "_raw_preview": (choice.content or "")[:1000]}

# ================= Generic Post-validation / Repair =================
GENERIC_TITLES = {
    "מתכון", "מתכון ביתי", "מתכון טעים", "מתכון להחזרת טעמים",
    "(ללא כותרת)", "Untitled", "Recipe"
}
GENERIC_INGREDIENTS = {"תבלינים", "תבלין", "מלח", "פלפל", "שמן", "שמן זית", "מים"}
MIN_INGREDIENTS = 3
MIN_STEPS = 2

# Optional, can stay empty. Add domain knowledge gradually.
DISH_RULES = {
    # Example (optional):
    # "בולונז": {
    #     "must_contain_any": {"עגבניות","רסק","passata","פסטה","גזר","סלרי","בצל","יין","חלב","שמנת","ragù","bolognese"}
    # }
}

def _text_blob_for_checks(r: dict, meta: dict | None) -> str:
    parts = [
        r.get("raw", {}).get("transcript", ""),
        json.dumps(r.get("evidence", {}), ensure_ascii=False),
    ]
    if meta:
        parts.append(meta.get("title",""))
        parts.append(meta.get("description",""))
        parts.extend(meta.get("tags", []))
    return " ".join(p for p in parts if p)

def _has_any(tokens: set[str], blob: str) -> bool:
    b = blob.lower()
    return any(t.lower() in b for t in tokens)

def validate_and_repair(r: dict, meta: dict | None = None) -> dict:
    if not isinstance(r, dict):
        return r

    # 1) Evidence gate: keep only ingredients that have evidence (if evidence exists)
    ev = r.get("evidence") or {}
    spans = ev.get("ingredient_spans") or []
    if spans:
        proven = {str(e.get("item","")).strip() for e in spans if isinstance(e, dict)}
        r["ingredients"] = [
            ing for ing in r.get("ingredients", [])
            if str(ing.get("item","")).strip() in proven
        ]

    # 2) Drop “generic-only” ingredients without quantities/notes
    cleaned = []
    for ing in r.get("ingredients", []):
        name = (ing.get("item") or "").strip()
        if name in GENERIC_INGREDIENTS and not ing.get("quantity") and not ing.get("notes"):
            continue
        cleaned.append(ing)
    r["ingredients"] = cleaned

    # 3) Title hygiene (prefer metadata title if model gave a generic one)
    title = (r.get("title") or "").strip()
    if title in GENERIC_TITLES:
        if meta and meta.get("title"):
            mt = meta["title"].strip()
            r["title"] = mt[:60] + ("…" if len(mt) > 60 else "")
            r["title_source"] = r.get("title_source") or "metadata"

    # 4) Universal minimums → lower confidence if too thin
    try:
        conf = float(r.get("confidence", 0.7))
    except Exception:
        conf = 0.7

    if len(r.get("ingredients", [])) < MIN_INGREDIENTS:
        conf = min(conf, 0.45)
    if len(r.get("steps", [])) < MIN_STEPS:
        conf = min(conf, 0.45)

    # 5) Title alignment: if we claim a dish name, ensure evidence exists in transcript/metadata
    blob = _text_blob_for_checks(r, meta)
    dish = (r.get("dish_canonical") or "").strip()
    title_terms = set((r.get("evidence", {}) or {}).get("title_terms", []))

    if dish and not (_has_any({dish} | title_terms, blob)):
        conf = min(conf, 0.40)

    # 6) Optional dish-specific rules (pluggable; safe to keep empty)
    rules = DISH_RULES.get(dish)
    if rules and "must_contain_any" in rules:
        if not _has_any(set(rules["must_contain_any"]), blob):
            conf = min(conf, 0.35)

    # 7) Always enforce language + source_url presence
    r["lang"] = "he"
    r["source_url"] = r.get("source_url") or None
    r["confidence"] = conf
    return r

# ================= Pretty-print =================
def pretty_recipe(r: dict) -> str:
    title = r.get("title") or "(ללא כותרת)"
    lines = [f"*{title}*"]
    if r.get("servings"): lines.append(f"_מנות_: {r['servings']}")
    if r.get("total_time_minutes"): lines.append(f"_זמן כולל_: {r['total_time_minutes']} דק'")
    if r.get("dish_canonical"): lines.append(f"_מנה_: {r['dish_canonical']}")

    lines.append("\n*מרכיבים*")
    for ing in r.get("ingredients", []):
        item = ing.get("item","")
        q = ing.get("quantity")
        u = ing.get("unit")
        qty = ""
        if q is not None and u: qty = f"{q:g} {u}"
        elif q is not None: qty = f"{q:g}"
        elif u: qty = u
        line = f"• {item}" + (f" — {qty}" if qty else "")
        if ing.get("notes"): line += f" ({ing['notes']})"
        lines.append(line)

    lines.append("\n*שלבים*")
    for s in r.get("steps", []):
        t = f" (~{s['time_minutes']} דק')" if s.get("time_minutes") else ""
        lines.append(f"{s.get('number', '?')}. {s.get('instruction','')}{t}")

    if r.get("notes"):
        lines.append("\n*הערות*")
        lines += [f"• {n}" for n in r["notes"]]

    conf = r.get("confidence")
    if conf is not None:
        try:
            lines.append(f"\n_Confidence_: {float(conf):.2f}")
        except Exception:
            pass

    if r.get("title_source"):
        lines.append(f"_מקור הכותרת_: {r['title_source']}")

    return "\n".join(lines)

# ================= Handlers =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "שלום! כאן המתכוניסט 👨🏻‍🍳, אפשר לשלוח לי קישור של סרטון אינסטגרם, טיקטוק או פייסבוק של מתכון, אני אוריד אתמלל ואשלח לך מתכון מוכן! שננסה?"
    )

async def router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.text:
        await msg.reply_text("קיבלתי הודעה שאינה טקסט. נסו לשלוח קישור.")
        return

    urls = find_urls(msg.text)
    if not urls:
        await msg.reply_text("(לא זוהה קישור) אמרתם: " + msg.text)
        return

    url = urls[0]
    url = canonicalize_url(url)   

    ok, reason = is_supported(url)
    if not ok:
        await msg.reply_text(f"⚠️ {reason}\n{url}")
        return

    await msg.reply_text("✅ הקישור תקין. מוריד ← מתמלל ← מחלץ מתכון…")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        try:
            # 1) Download & extract audio (blocking → offload)
            got = await asyncio.to_thread(download_media, url, workdir)

            # 2) Transcribe
            transcript = await asyncio.to_thread(transcribe_audio, got["audio_path"])

            # 3) Recipe JSON (LLM) with metadata hints
            recipe = await asyncio.to_thread(make_recipe, transcript, got["source_url"], got["meta"])
            if "_parse_error" in recipe:
                await msg.reply_text(
                    "⚠️ נוצרה תשובה אך נכשל פענוח JSON.\n"
                    f"שגיאת מפענח: {recipe['_parse_error']}\n"
                    "אפשר לנסות שוב, ואם לא עובד, אפשר לדווח  omerkrespi.1@gmail.com"
                )
                await msg.reply_text(recipe.get("_raw_preview","")[:1000])
                return

            # 4) Post-validate/repair (generic, metadata-aware)
            recipe = validate_and_repair(recipe, meta=got.get("meta"))

            # 5) Human-friendly message
            pretty = pretty_recipe(recipe)
            await msg.reply_text(pretty, parse_mode="Markdown")

            # 6) Attach recipe.json
            json_path = workdir / "recipe.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(recipe, f, ensure_ascii=False, indent=2)
            with open(json_path, "rb") as f:
                await msg.reply_document(InputFile(f, filename="recipe.json"))

        except Exception as e:
            await msg.reply_text(f"❌ Error: {e}")
            logger.exception("Pipeline error")

def main() -> None:
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, router))
    logger.info("Bot polling…")
    app.run_polling()

if __name__ == "__main__":
    main()
