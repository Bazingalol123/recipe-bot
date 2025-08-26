FROM python:3.11-slim

# ffmpeg for audio extraction
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt

# copy app code (must include lambda_handler.py and app_pipeline.py)
COPY . .
RUN python -m py_compile app_pipeline.py lambda_handler.py
# copy cookies file and set env var for yt-dlp to use it
# COPY cookies.txt  /var/task/cookies.txt
# ENV YTDLP_COOKIES_FILE=/var/task/cookies.txt


# ✅ Correct way for Lambda container images:
ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["lambda_handler.lambda_handler"]
