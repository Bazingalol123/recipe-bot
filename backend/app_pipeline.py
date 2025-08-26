# -*- coding: utf-8 -*-
# app_pipeline.py — full pipeline used by Lambda webhook

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
import os, re, json, subprocess, tempfile, logging, time, shlex, shutil
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from openai import OpenAI
import boto3

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# ===== Config =====
LLM_MODEL = "gpt-4o-mini"
ASR_MODEL = "gpt-4o-mini-transcribe"
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
SUPPORTED = ("instagram.com", "tiktok.com", "facebook.com", "fb.watch")
IG_MEDIA_PATH = re.compile(r"^/(reel|reels|p|tv)/([A-Za-z0-9_\-]+)/?")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------- URL helpers ----------
def find_first_url(text: str) -> str | None:
    logger.debug("Searching for first URL in text: %r", text)
    m = URL_RE.search(text or "")
    url = m.group(0) if m else None
    logger.info("Found URL: %r", url)
    return url

def canonicalize_url(u: str) -> str:
    """Normalize social links so yt-dlp behaves consistently."""
    logger.debug("Canonicalizing URL: %r", u)
    p = urlparse(u)
    host = (p.netloc or "").lower()
    scheme = p.scheme or "https"

    # Instagram → keep only /{kind}/{id}/
    if "instagram.com" in host:
        m = IG_MEDIA_PATH.match(p.path)
        if m:
            kind, media_id = m.groups()
            result = urlunparse((scheme, "www.instagram.com", f"/{kind}/{media_id}/", "", "", ""))
            logger.info("Canonicalized Instagram URL: %r", result)
            return result
        result = urlunparse((scheme, "www.instagram.com", p.path, "", "", ""))
        logger.info("Canonicalized Instagram URL: %r", result)
        return result

    # TikTok → keep path, drop query/fragment
    if "tiktok.com" in host:
        path = p.path.rstrip("/") or "/"
        result = urlunparse((scheme, "www.tiktok.com", path + ("/" if not path.endswith("/") else ""), "", "", ""))
        logger.info("Canonicalized TikTok URL: %r", result)
        return result

    # Facebook → drop trackers, keep meaningful query like v=...
    if "facebook.com" in host or "fb.watch" in host:
        q = dict(parse_qsl(p.query))
        for k in list(q.keys()):
            if k.startswith("utm_") or k in {"mibextid", "s", "si"}:
                q.pop(k, None)
        qstr = urlencode(q) if q else ""
        result = urlunparse((scheme, host, p.path, "", qstr, ""))
        logger.info("Canonicalized Facebook URL: %r", result)
        return result

    logger.info("URL did not match any canonicalization rules: %r", u)
    return u

# ---------- Shell helpers ----------
def run_cmd(cmd: str) -> str:
    logger.info("Running command: %s", cmd)
    try:
        p = subprocess.run(
            cmd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        logger.debug("Command output: %s", p.stdout)
        logger.debug("Command error: %s", p.stderr)
        return (p.stdout or "") + (p.stderr or "")
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s\n%s", cmd, e.stderr)
        raise

# ---------- COOKIES (S3 → /tmp/cookies.txt, with local fallback) ----------
def get_cookie_file_from_s3() -> str | None:
    """
    Materialize a writable cookies file for yt-dlp at /tmp/ig_cookies.txt (0600).
    Source: IG_COOKIES_S3 (s3://...) or local fallback YTDLP_COOKIES_FILE.
    """
    s3_uri = os.getenv("IG_COOKIES_S3")
    dst = "/tmp/ig_cookies.txt"

    # Clean previous file if exists (avoid perms surprises)
    try:
        if Path(dst).exists():
            os.remove(dst)
    except Exception:
        pass

    # Local fallback for dev
    fallback = os.getenv("YTDLP_COOKIES_FILE")
    if (not s3_uri) and fallback and Path(fallback).exists():
        shutil.copyfile(fallback, dst)
    elif s3_uri and s3_uri.startswith("s3://"):
        bucket, key = s3_uri[5:].split("/", 1)
        boto3.client("s3").download_file(bucket, key, dst)
    else:
        return None

    # Ensure owner-writable so yt-dlp can save back
    try:
        os.chmod(dst, 0o600)
    except Exception:
        pass

    return dst if Path(dst).exists() else None


def summarize_ig_cookies(path: str) -> bool:
    """
    Log basic health of a Netscape cookies file; return True if it looks usable.
    Checks for 7 tab-separated columns, required names, and expiry.
    """
    try:
        lines = Path(path).read_text('utf-8', errors='ignore').splitlines()
    except FileNotFoundError:
        logger.warning("[IG COOKIES] file not found: %s", path)
        return False
    rows = [l for l in lines if l and not l.startswith('#')]
    bad = [i for i, l in enumerate(rows, 1) if len(l.split('\t')) != 7]
    if bad:
        logger.warning("[IG COOKIES] wrong column count on lines: %s", bad[:10])
    present, expired = set(), []
    now = int(time.time())
    names = []
    for l in rows:
        parts = l.split('\t')
        if len(parts) != 7:
            continue
        _, _, _, _, expiry, name, _ = parts
        names.append(name)
        if name in ('sessionid', 'csrftoken', 'ds_user_id'):
            present.add(name)
        try:
            if expiry not in ('', '0') and int(expiry) < now:
                expired.append(name)
        except Exception:
            pass
    logger.info("[IG COOKIES] found %d cookies; sample: %s", len(names), ','.join(names[:8]))
    missing = {'sessionid', 'csrftoken', 'ds_user_id'} - present
    if missing:
        logger.warning("[IG COOKIES] missing: %s", ', '.join(sorted(missing)))
    if expired:
        logger.warning("[IG COOKIES] expired: %s", ', '.join(sorted(set(expired))))
    return not missing and not expired and not bad

# ---------- Download + metadata + audio ----------
def download_media(url: str, workdir: Path) -> dict:
    """
    Download Instagram/TikTok/Facebook video, extract metadata, and audio.
    Returns dict with audio_path, source_url, meta.
    """
    UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    host = (urlparse(url).netloc or "").lower()
    out_tmpl = str(workdir / "vid.%(ext)s")
    cookies = get_cookie_file_from_s3()
    has_cookies = bool(cookies and summarize_ig_cookies(cookies))

    # ---- download video via yt-dlp API ----
    logger.info("[IGDL] Starting Instagram download for url=%r", url)
    def _ig_headers(ua: str) -> dict:
        return {"User-Agent": ua, "Referer": "https://www.instagram.com/"}
    
    ydl_dl_opts = {
        "outtmpl": out_tmpl,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "concurrent_fragment_downloads": 1,
        "cookiefile": cookies if has_cookies else None,
        "quiet": True,
        "no_warnings": True,
    }
    if "instagram.com" in host:
        ydl_dl_opts["http_headers"] = _ig_headers(UA)
    
    try:
        logger.info("[IGDL] yt-dlp download options: %r", ydl_dl_opts)
        with YoutubeDL({k: v for k, v in ydl_dl_opts.items() if v is not None}) as ydl:
            ydl.download([url])
        logger.info("[IGDL] Download completed for url=%r", url)
    except DownloadError as e:
        logger.error("[IGDL] yt-dlp download failed: %s", e)
        raise
    
    # find downloaded file
    video_path = next(
        (f for f in workdir.iterdir()
         if f.name.startswith("vid.") and f.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")),
        None
    )
    logger.info("[IGDL] Downloaded video path: %r", video_path)
    if not video_path:
        logger.error("[IGDL] Could not find downloaded video file in %s", workdir)
        raise RuntimeError("Could not find downloaded video file.")
    
    # ---- metadata via yt-dlp API ----
    logger.info("[IGDL] Starting metadata extraction for url=%r", url)
    ydl_meta_opts = {
        "skip_download": True,
        "cookiefile": cookies if has_cookies else None,
        "quiet": True,
        "no_warnings": True,
    }
    if "instagram.com" in host:
        ydl_meta_opts["http_headers"] = _ig_headers(UA)
    
    meta: dict = {}
    try:
        logger.info("[IGDL] yt-dlp metadata options: %r", ydl_meta_opts)
        with YoutubeDL({k: v for k, v in ydl_meta_opts.items() if v is not None}) as ydl:
            info = ydl.extract_info(url, download=False) or {}
            meta = {
                "title": info.get("title") or "",
                "description": info.get("description") or "",
                "tags": info.get("tags") or [],
            }
        logger.info("[IGDL] Metadata extraction succeeded: %r", meta)
    except Exception as e:
        logger.warning("[IGDL] Metadata extraction failed, trying fallback to write infojson: %s", e)
        try:
            # Ask yt-dlp to write info json alongside out_tmpl
            ydl_fb_opts = {
                "skip_download": True,
                "writeinfojson": True,
                "outtmpl": out_tmpl,
                "cookiefile": cookies if has_cookies else None,
                "quiet": True,
                "no_warnings": True,
            }
            if "instagram.com" in host:
                ydl_fb_opts["http_headers"] = _ig_headers(UA)
            logger.info("[IGDL] yt-dlp fallback metadata options: %r", ydl_fb_opts)
            with YoutubeDL({k: v for k, v in ydl_fb_opts.items() if v is not None}) as ydl:
                ydl.extract_info(url, download=False)
    
            info_file = max(
                list(workdir.glob("*.info.json")) or list(workdir.glob("*.json")),
                default=None,
                key=lambda p: p.stat().st_mtime if p else 0
            )
            logger.info("[IGDL] Fallback info file: %r", info_file)
            if info_file:
                mj = json.loads(info_file.read_text(encoding="utf-8", errors="ignore"))
                meta = {
                    "title": mj.get("title") or "",
                    "description": mj.get("description") or "",
                    "tags": mj.get("tags") or [],
                }
                logger.info("[IGDL] Fallback metadata extraction succeeded: %r", meta)
        except Exception as e2:
            logger.error("[IGDL] Fallback metadata extraction failed: %s", e2)
            meta = {}

    # ---- extract audio ----
    audio_path = str(workdir / "audio.mp3")
    try:
        run_cmd(f'ffmpeg -y -i "{video_path}" -vn -acodec libmp3lame -ar 44100 -ac 2 -ab 192k -f mp3 "{audio_path}"')
        logger.info("[IGDL] Audio extraction succeeded: %s", audio_path)
    except Exception as e:
        logger.error("[IGDL] Audio extraction failed: %s", e)
        raise

    return {
        "audio_path": audio_path,
        "source_url": url,
        "meta": meta,
    }

# ---------- Transcription ----------
def transcribe(audio_path: str) -> str:
    """Prefer Hebrew; if it fails, retry autodetect."""
    logger.info("Transcribing audio: %s", audio_path)
    with open(audio_path, "rb") as f:
        try:
            resp = client.audio.transcriptions.create(model=ASR_MODEL, file=f, language="he")
            logger.info("Transcription (hebrew) succeeded")
            return resp.text
        except Exception as e:
            logger.warning("Transcription (hebrew) failed: %s, retrying autodetect", e)
            f.seek(0)
            resp2 = client.audio.transcriptions.create(model=ASR_MODEL, file=f)
            logger.info("Transcription (autodetect) succeeded")
            return resp2.text

# ---------- JSON helpers ----------
def safe_load_json(s: str) -> dict:
    logger.debug("Attempting to safely load JSON")
    s = (s or "").strip()
    # strip markdown fences and leading "json" marker
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    # normalize smart quotes to straight quotes
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    # drop control chars that often break json parsing
    s = re.sub(r'[\x00-\x1f\x7f]', '', s)
    # try to extract the largest {...} block if present
    matches = re.findall(r'\{.*\}', s, flags=re.S)
    if matches:
        s = max(matches, key=len)
    else:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1]
    # common fixes: remove trailing commas before } or ]
    s_fixed = re.sub(r',\s*(?=[}\]])', '', s)
    # collapse accidental repeated commas
    s_fixed = re.sub(r',\s*,+', ',', s_fixed)
    try:
        return json.loads(s_fixed)
    except Exception as e1:
        logger.warning("safe_load_json initial parse failed: %s; attempting softer fallbacks", e1)
        # fallback: try adding escaped newlines and reparse
        try:
            maybe = s_fixed.replace('\n', '\\n').replace('\r', '')
            return json.loads(maybe)
        except Exception as e2:
            logger.error("Failed to parse JSON after fallbacks: %s", e2)
            return {"_parse_error": str(e2), "_raw_preview": s[:4000]}

RECIPE_TOOL_PARAMETERS = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "title": {"type": "string"},
    "source_url": {"type": "string"},
    "lang": {"type": "string", "enum": ["he"]},
    "servings": {"type": ["integer", "null"]},
    "total_time_minutes": {"type": ["integer", "null"]},
    "ingredients": {"type": "array", "items": {
      "type": "object", "additionalProperties": False,
      "properties": {
        "item": {"type": "string"},
        "quantity": {"type": ["number", "null"]},
        "unit": {"type": ["string", "null"], "enum": ["גרם","מ״ל","כפית","כף","כוס","יחידה"]},
        "notes": {"type": ["string", "null"]}
      }, "required": ["item"]
    }},
    "steps": {"type": "array", "items": {
      "type": "object", "additionalProperties": False,
      "properties": {
        "number": {"type": "integer"},
        "instruction": {"type": "string"},
        "time_minutes": {"type": ["integer", "null"]}
      }, "required": ["number","instruction"]
    }},
    "equipment": {"type": "array", "items": {"type": "string"}},
    "notes": {"type": "array", "items": {"type": "string"}},
    "nutrition_estimate": {"type": "object", "additionalProperties": False,
      "properties": {"kcal_per_serving": {"type": ["number","null"]}}
    },
    "confidence": {"type": "number"},
    "raw": {"type": "object", "additionalProperties": False,
      "properties": {"transcript": {"type": "string"}}, "required": ["transcript"]
    },
    "dish_canonical": {"type": ["string","null"]},
    "title_source": {"type": ["string","null"], "enum": ["transcript","metadata","both"]},
    "evidence": {"type": "object", "additionalProperties": False, "properties": {
      "title_terms": {"type":"array","items":{"type":"string"}},
      "ingredient_spans": {"type":"array","items":{
        "type":"object","properties":{"item":{"type":"string"},"snippet":{"type":"string"}},
        "required":["item","snippet"], "additionalProperties": False
      }}
    }}
  },
  "required": ["title","ingredients","steps","confidence","raw","lang"]
}

SYSTEM_PROMPT = (
  "אתה מנוע חילוץ קולינרי. ענה בעברית בלבד.\n"
  "השתמש אך ורק בתמלול וב-metadata שסופקו (כותרת/תיאור/תגיות). "
  "אסור להמציא מרכיבים או שלבים שאינם מופיעים במפורש. אם חסר מידע – השאר null/ריק והורד confidence. "
  "נרמל יחידות ל: גרם, מ״ל, כפית, כף, כוס, יחידה. הגדר lang='he'. אין לכלול אותיות לטיניות בפלט.\n"
  "כותרת: הימנע מכותרות כלליות. אם בתמלול או ב-metadata מופיעים שמות מנה (למשל 'בולונז' / 'bolognese' / 'ragù'), "
  "עדכן title בהתאם, הגדר dish_canonical לשם המנה, ו-title_source='transcript'/'metadata'/'both'. "
  "החזר evidence: לכל מרכיב ציין snippet מהתמלול/metadata שמצדיק אותו; וכן רשימת מילות מפתח שהשפיעו על הכותרת."
)

def _parse_tool_args(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return safe_load_json(s)

def make_recipe(transcript: str, source_url: str, meta: dict) -> dict:
    logger.info("Making recipe from transcript and metadata")
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

    def _call(msgs):
        return client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            messages=msgs,
            tools=tools,
            tool_choice={"type":"function","function":{"name":"submit_recipe"}},
            parallel_tool_calls=False,
            max_tokens=3000,
        )

# First attempt
    resp = _call(messages)
    logger.warning("LLM raw response (first attempt): %r", resp)
    r = resp.choices[0].message
    if getattr(r, "tool_calls", None):
        logger.info("LLM returned tool call")
        data = _parse_tool_args(r.tool_calls[0].function.arguments)
        if "_parse_error" not in data:
            data.setdefault("raw", {})["transcript"] = transcript[:1200]
            data["source_url"] = source_url
            data["lang"] = "he"
            return data

    # Repair attempt
    logger.warning("First attempt failed, trying repair")
    repair = [
        {"role":"system","content":
         "הארגומנטים לכלי לא היו JSON תקין. החזר אך ורק קריאה ל-submit_recipe עם JSON חוקי. "
         "ללא טקסט חופשי. שמור raw.transcript ≤ 1200."},
        {"role":"user","content": json.dumps({
            "source_url": source_url, "metadata": meta, "transcript": transcript[:3500]
        }, ensure_ascii=False)}
    ]
    resp2 = _call(repair)
    logger.warning("LLM raw response (repair attempt): %r", resp2)
    r2 = resp2.choices[0].message
    logger.warning("LLM repair raw response: %r", getattr(r2, "content", None))
    if getattr(r2, "tool_calls", None):
        logger.info("Repair attempt succeeded")
        data2 = _parse_tool_args(r2.tool_calls[0].function.arguments)
        data2.setdefault("raw", {})["transcript"] = transcript[:1200]
        data2["source_url"] = source_url
        data2["lang"] = "he"
        return data2

    logger.error("No tool call returned from LLM")
    logger.warning("LLM repair raw response (no tool call): %r", getattr(r2, "content", None))
    return {"_parse_error":"No tool call returned", "_raw_preview": (r.content or "")[:1000]}

# ---------- Post-validation / repair ----------
GENERIC_TITLES = {"מתכון","מתכון ביתי","מתכון טעים","(ללא כותרת)","Untitled","Recipe"}
GENERIC_INGREDIENTS = {"תבלינים","תבלין","מלח","פלפל","שמן","שמן זית","מים"}
MIN_INGREDIENTS, MIN_STEPS = 3, 2
DISH_RULES = {
    # "בולונז": {"must_contain_any": {"עגבניות","רסק","passata","פסטה","גזר","סלרי","בצל","יין","ragù","bolognese"}}
}

def _blob(r: dict, meta: dict | None) -> str:
    parts = [r.get("raw",{}).get("transcript",""), json.dumps(r.get("evidence",{}), ensure_ascii=False)]
    if meta:
        parts += [meta.get("title",""), meta.get("description","")] + list(meta.get("tags",[]))
    return " ".join(p for p in parts if p)

def _has_any(tokens: set[str], blob: str) -> bool:
    b = blob.lower()
    return any(t.lower() in b for t in tokens)

def validate_and_repair(r: dict, meta: dict | None = None) -> dict:
    logger.info("Validating and repairing recipe")
    if not isinstance(r, dict): return r

    ev = r.get("evidence") or {}
    spans = ev.get("ingredient_spans") or []
    if spans:
        proven = {str(e.get("item","")).strip() for e in spans if isinstance(e, dict)}
        r["ingredients"] = [ing for ing in r.get("ingredients", [])
                            if str(ing.get("item","")).strip() in proven]

    cleaned = []
    for ing in r.get("ingredients", []):
        name = (ing.get("item") or "").strip()
        if name in GENERIC_INGREDIENTS and not ing.get("quantity") and not ing.get("notes"):
            continue
        cleaned.append(ing)
    r["ingredients"] = cleaned

    title = (r.get("title") or "").strip()
    if title in GENERIC_TITLES and meta and meta.get("title"):
        mt = meta["title"].strip()
        r["title"] = mt[:60] + ("…" if len(mt) > 60 else "")
        r["title_source"] = r.get("title_source") or "metadata"

    try: conf = float(r.get("confidence", 0.7))
    except Exception: conf = 0.7
    if len(r.get("ingredients", [])) < MIN_INGREDIENTS: conf = min(conf, 0.45)
    if len(r.get("steps", [])) < MIN_STEPS: conf = min(conf, 0.45)

    blob = _blob(r, meta)
    dish = (r.get("dish_canonical") or "").strip()
    title_terms = set((r.get("evidence",{}) or {}).get("title_terms", []))
    if dish and not (_has_any({dish} | title_terms, blob)):
        conf = min(conf, 0.40)

    rules = DISH_RULES.get(dish)
    if rules and "must_contain_any" in rules:
        if not _has_any(set(rules["must_contain_any"]), blob):
            conf = min(conf, 0.35)

    r["lang"] = "he"
    r["source_url"] = r.get("source_url") or None
    r["confidence"] = conf
    logger.info("Validation complete, confidence: %s", r.get("confidence"))
    return r

# ---------- Pretty-print ----------
def pretty_recipe(r: dict) -> str:
    logger.debug("Formatting recipe for pretty print")
    title = r.get("title") or "(ללא כותרת)"
    lines = [f"*{title}*"]
    if r.get("servings"): lines.append(f"_מנות_: {r['servings']}")
    if r.get("total_time_minutes"): lines.append(f"_זמן כולל_: {r['total_time_minutes']} דק'")
    if r.get("dish_canonical"): lines.append(f"_מנה_: {r['dish_canonical']}")

    lines.append("\n*מרכיבים*")
    for ing in r.get("ingredients", []):
        item = ing.get("item","")
        q, u = ing.get("quantity"), ing.get("unit")
        qty = (f"{q:g} {u}" if (q is not None and u) else f"{q:g}" if q is not None else (u or "")).strip()
        line = f"• {item}" + (f" — {qty}" if qty else "")
        if ing.get("notes"): line += f" ({ing['notes']})"
        lines.append(line)

    lines.append("\n*שלבים*")
    for s in r.get("steps", []):
        t = f" (~{s['time_minutes']} דק')" if s.get("time_minutes") else ""
        lines.append(f"{s.get('number','?')}. {s.get('instruction','')}{t}")

    if r.get("notes"):
        lines.append("\n*הערות*")
        lines += [f"• {n}" for n in r["notes"]]

    conf = r.get("confidence")
    if conf is not None:
        try: lines.append(f"\n_Confidence_: {float(conf):.2f}")
        except: pass
    if r.get("title_source"):
        lines.append(f"_מקור הכותרת_: {r['title_source']}")
    return "\n".join(lines)

# ---------- Orchestrator ----------
def process_link(text: str) -> tuple[str, bytes]:
    logger.info("Processing link from text")
    url = find_first_url(text)
    if not url or not any(root in url.lower() for root in SUPPORTED):
        logger.error("No supported link found in message: %r", text)
        raise ValueError("No supported link found in your message.")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        logger.info("Working directory: %s", work)
        got = download_media(url, work)
        transcript = transcribe(got["audio_path"])
        recipe = make_recipe(transcript, got["source_url"], got.get("meta") or {})
        if "_parse_error" in recipe:
            logger.error("Recipe JSON parse error: %s", recipe["_parse_error"])
            raise RuntimeError("Recipe JSON parse error: " + recipe["_parse_error"])
        recipe = validate_and_repair(recipe, got.get("meta"))
        pretty = pretty_recipe(recipe)
        logger.info("Recipe processing complete")
        return pretty, json.dumps(recipe, ensure_ascii=False, indent=2).encode("utf-8")