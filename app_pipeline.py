# app_pipeline.py
import os, re, json, subprocess, tempfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from openai import OpenAI

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
SUPPORTED = ("instagram.com", "tiktok.com", "facebook.com", "fb.watch")

def find_first_url(text: str) -> str | None:
    m = URL_RE.search(text or "")
    return m.group(0) if m else None

def canonicalize_url(u: str) -> str:
    p = urlparse(u)
    host = (p.netloc or "").lower()
    scheme = p.scheme or "https"

    # Instagram: force canonical /{kind}/{id}/ and drop tracking
    if "instagram.com" in host:
        m = re.match(r"^/(reel|reels|p|tv)/([A-Za-z0-9_\-]+)/?", p.path)
        if m:
            kind, media_id = m.groups()
            return urlunparse((scheme, "www.instagram.com", f"/{kind}/{media_id}/", "", "", ""))
        return urlunparse((scheme, "www.instagram.com", p.path, "", "", ""))

    # TikTok: drop query/fragment, keep path
    if "tiktok.com" in host:
        path = p.path.rstrip("/") or "/"
        return urlunparse((scheme, "www.tiktok.com", path + ("/" if not path.endswith("/") else ""), "", "", ""))

    # Facebook: keep meaningful query (e.g., v=...), drop trackers
    if "facebook.com" in host or "fb.watch" in host:
        q = dict(parse_qsl(p.query))
        for k in list(q.keys()):
            if k.startswith("utm_") or k in {"mibextid", "s", "si"}:
                q.pop(k, None)
        qstr = urlencode(q) if q else ""
        return urlunparse((scheme, host, p.path, "", qstr, ""))

    return u

def run_cmd(args: list[str]) -> str:
    # safer than shell=True; returns stdout+stderr text; raises on error
    res = subprocess.run(args, check=True, capture_output=True, text=True)
    return (res.stdout or "") + (res.stderr or "")

def download_media(url: str, workdir: Path) -> tuple[Path, Path, str]:
    """Download a video with yt-dlp and extract mono 16k wav with ffmpeg."""
    url = canonicalize_url(url)
    out_tmpl = str(workdir / "vid.%(ext)s")

    # add a desktop UA (helps with IG)
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
        run_cmd(args)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed.\n{(e.stderr or str(e))[:1000]}")

    video_path = next((f for f in workdir.iterdir()
                       if f.name.startswith("vid.") and f.suffix.lower() in (".mp4",".mkv",".webm",".mov")), None)
    if not video_path:
        raise RuntimeError("Downloaded video not found.")

    audio_path = workdir / "audio.wav"
    run_cmd(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)])
    return video_path, audio_path, url

def transcribe(audio_path: Path) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
    return resp.text

RECIPE_SCHEMA = {
  "type":"object",
  "properties":{
    "title":{"type":"string"},
    "source_url":{"type":"string"},
    "lang":{"type":"string"},
    "servings":{"type":["integer","null"]},
    "total_time_minutes":{"type":["integer","null"]},
    "ingredients":{"type":"array","items":{
      "type":"object","properties":{
        "item":{"type":"string"},
        "quantity":{"type":["number","null"]},
        "unit":{"type":["string","null"]},
        "notes":{"type":["string","null"]}
      },"required":["item"]
    }},
    "steps":{"type":"array","items":{
      "type":"object","properties":{
        "number":{"type":"integer"},
        "instruction":{"type":"string"},
        "time_minutes":{"type":["integer","null"]}
      },"required":["number","instruction"]
    }},
    "equipment":{"type":"array","items":{"type":"string"}},
    "notes":{"type":"array","items":{"type":"string"}},
    "nutrition_estimate":{"type":"object","properties":{
      "kcal_per_serving":{"type":["number","null"]}
    }},
    "confidence":{"type":"number"},
    "raw":{"type":"object","properties":{"transcript":{"type":"string"}}}
  },
  "required":["title","ingredients","steps","confidence","raw"]
}

def safe_load_json(s: str) -> dict:
    s = s.strip()
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

def make_recipe(transcript: str, source_url: str) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system = ("You are a culinary extraction engine. "
              "Given a cooking video transcript, output JSON ONLY in the provided schema. "
              "Infer servings/time if hinted. Normalize units. Include confidence 0..1. "
              "Keep raw.transcript under ~4000 chars.")
    user = {"schema": RECIPE_SCHEMA, "source_url": source_url, "transcript": transcript[:4000]}
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":system},
                  {"role":"user","content":json.dumps(user)}],
        max_tokens=1200,
    )
    return safe_load_json(resp.choices[0].message.content or "")

def pretty_recipe(r: dict) -> str:
    title = r.get("title") or "(Untitled)"
    lines = [f"*{title}*"]
    if r.get("servings"): lines.append(f"_Servings_: {r['servings']}")
    if r.get("total_time_minutes"): lines.append(f"_Total time_: {r['total_time_minutes']} min")
    lines.append("\n*Ingredients*")
    for ing in r.get("ingredients", []):
        item = ing.get("item","")
        q, u = ing.get("quantity"), ing.get("unit")
        qty = (f"{q:g} {u}" if (q is not None and u) else f"{q:g}" if q is not None else (u or "")).strip()
        line = f"• {item}" + (f" — {qty}" if qty else "")
        if ing.get("notes"): line += f" ({ing['notes']})"
        lines.append(line)
    lines.append("\n*Steps*")
    for s in r.get("steps", []):
        t = f" (~{s['time_minutes']} min)" if s.get("time_minutes") else ""
        lines.append(f"{s.get('number','?')}. {s.get('instruction','')}{t}")
    if r.get("notes"):
        lines.append("\n*Notes*")
        lines += [f"• {n}" for n in r["notes"]]
    conf = r.get("confidence")
    if conf is not None:
        try: lines.append(f"\n_Confidence_: {float(conf):.2f}")
        except: pass
    return "\n".join(lines)

def process_link(text: str) -> tuple[str, bytes]:
    url = find_first_url(text)
    if not url or not any(root in url.lower() for root in SUPPORTED):
        raise ValueError("No supported link found in your message.")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        video, audio, clean_url = download_media(url, work)
        transcript = transcribe(audio)
        recipe = make_recipe(transcript, clean_url)
        if "_parse_error" in recipe:
            raise RuntimeError("Recipe JSON parse error: " + recipe["_parse_error"])
        pretty = pretty_recipe(recipe)
        return pretty, json.dumps(recipe, ensure_ascii=False, indent=2).encode("utf-8")
