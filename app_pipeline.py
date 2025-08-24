# -*- coding: utf-8 -*-
# app_pipeline.py — full pipeline used by Lambda webhook
import os, re, json, subprocess, tempfile, logging
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from openai import OpenAI

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
    m = URL_RE.search(text or "")
    return m.group(0) if m else None

def canonicalize_url(u: str) -> str:
    """Normalize social links so yt-dlp behaves consistently."""
    p = urlparse(u)
    host = (p.netloc or "").lower()
    scheme = p.scheme or "https"

    # Instagram → keep only /{kind}/{id}/
    if "instagram.com" in host:
        m = IG_MEDIA_PATH.match(p.path)
        if m:
            kind, media_id = m.groups()
            return urlunparse((scheme, "www.instagram.com", f"/{kind}/{media_id}/", "", "", ""))
        return urlunparse((scheme, "www.instagram.com", p.path, "", "", ""))

    # TikTok → keep path, drop query/fragment
    if "tiktok.com" in host:
        path = p.path.rstrip("/") or "/"
        return urlunparse((scheme, "www.tiktok.com", path + ("/" if not path.endswith("/") else ""), "", "", ""))

    # Facebook → drop trackers, keep meaningful query like v=...
    if "facebook.com" in host or "fb.watch" in host:
        q = dict(parse_qsl(p.query))
        for k in list(q.keys()):
            if k.startswith("utm_") or k in {"mibextid", "s", "si"}:
                q.pop(k, None)
        qstr = urlencode(q) if q else ""
        return urlunparse((scheme, host, p.path, "", qstr, ""))

    return u

# ---------- Shell helpers ----------
def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Run a command (no shell), capture output, raise on non-zero."""
    return subprocess.run(args, check=True, capture_output=True, text=True)

# ---------- Download + metadata + audio ----------
def download_media(url: str, workdir: Path) -> dict:
    """Fetch best video with yt-dlp, extract mono 16k WAV, and collect basic metadata."""
    url = canonicalize_url(url)
    out_tmpl = str(workdir / "vid.%(ext)s")

    # Helpful UA for IG
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36")
    host = urlparse(url).netloc.lower()

    args = ["yt-dlp", "-o", out_tmpl, "--merge-output-format", "mp4", url]
    if "instagram.com" in host:
        args = ["yt-dlp", "--user-agent", UA, "--referer", "https://www.instagram.com/",
                "-o", out_tmpl, "--merge-output-format", "mp4", url]
        cookies = os.getenv("YTDLP_COOKIES_FILE")
        if cookies and Path(cookies).exists():
            args = ["yt-dlp", "--cookies", cookies,
                    "--user-agent", UA, "--referer", "https://www.instagram.com/",
                    "-o", out_tmpl, "--merge-output-format", "mp4", url]

    try:
        _run(args)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed.\n{(e.stderr or str(e))[:1000]}")

    # Find video
    video_path = next((f for f in workdir.iterdir()
                       if f.name.startswith("vid.") and f.suffix.lower() in (".mp4",".mkv",".webm",".mov")), None)
    if not video_path:
        raise RuntimeError("Downloaded video not found.")

    # Extract audio
    audio_path = workdir / "audio.wav"
    _run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)])

    # Metadata (best-effort)
    meta = {}
    try:
        j = _run(["yt-dlp", "-J", url]).stdout
        mj = json.loads(j)
        meta = {
            "title": mj.get("title") or "",
            "description": mj.get("description") or "",
            "tags": mj.get("tags") or [],
        }
    except Exception:
        meta = {}

    return {
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "source_url": url,
        "meta": meta,
    }

# ---------- Transcription ----------
def transcribe(audio_path: str) -> str:
    """Prefer Hebrew; if it fails, retry autodetect."""
    with open(audio_path, "rb") as f:
        try:
            resp = client.audio.transcriptions.create(model=ASR_MODEL, file=f, language="he")
            return resp.text
        except Exception:
            f.seek(0)
            resp2 = client.audio.transcriptions.create(model=ASR_MODEL, file=f)
            return resp2.text

# ---------- JSON helpers ----------
def safe_load_json(s: str) -> dict:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    try:
        return json.loads(s)
    except Exception as e:
        return {"_parse_error": str(e), "_raw_preview": s[:1000]}

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
  "אסור להמציא מרכיבים או שלבים. אם חסר מידע – השאר null/ריק והורד confidence. "
  "נרמל יחידות ל: גרם, מ״ל, כפית, כף, כוס, יחידה. הגדר lang='he'. אין לכלול אותיות לטיניות בפלט.\n"
  "כותרת: הימנע מכותרות כלליות. אם בתמלול או ב-metadata מופיע שם מנה, עדכן title בהתאם, "
  "קבע dish_canonical ו-title_source. החזר evidence: לכל מרכיב ציין snippet שמצדיק אותו, "
  "וכן רשימת מילות מפתח שהשפיעו על הכותרת."
)

def _parse_tool_args(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return safe_load_json(s)

def make_recipe(transcript: str, source_url: str, meta: dict) -> dict:
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
            max_tokens=1200,
        )

    # First attempt
    r = _call(messages).choices[0].message
    if getattr(r, "tool_calls", None):
        data = _parse_tool_args(r.tool_calls[0].function.arguments)
        if "_parse_error" not in data:
            data.setdefault("raw", {})["transcript"] = transcript[:1200]
            data["source_url"] = source_url
            data["lang"] = "he"
            return data

    # Repair attempt
    repair = [
        {"role":"system","content":
         "הארגומנטים לכלי לא היו JSON תקין. החזר אך ורק קריאה ל-submit_recipe עם JSON חוקי. "
         "ללא טקסט חופשי. שמור raw.transcript ≤ 1200."},
        {"role":"user","content": json.dumps({
            "source_url": source_url, "metadata": meta, "transcript": transcript[:3500]
        }, ensure_ascii=False)}
    ]
    r2 = _call(repair).choices[0].message
    if getattr(r2, "tool_calls", None):
        data2 = _parse_tool_args(r2.tool_calls[0].function.arguments)
        data2.setdefault("raw", {})["transcript"] = transcript[:1200]
        data2["source_url"] = source_url
        data2["lang"] = "he"
        return data2

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
    return r

# ---------- Pretty-print ----------
def pretty_recipe(r: dict) -> str:
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
    url = find_first_url(text)
    if not url or not any(root in url.lower() for root in SUPPORTED):
        raise ValueError("No supported link found in your message.")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        got = download_media(url, work)
        transcript = transcribe(got["audio_path"])
        recipe = make_recipe(transcript, got["source_url"], got.get("meta") or {})
        if "_parse_error" in recipe:
            raise RuntimeError("Recipe JSON parse error: " + recipe["_parse_error"])
        recipe = validate_and_repair(recipe, got.get("meta"))
        pretty = pretty_recipe(recipe)
        return pretty, json.dumps(recipe, ensure_ascii=False, indent=2).encode("utf-8")
