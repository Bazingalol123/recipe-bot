FROM python:3.11-slim

# ffmpeg for audio extraction
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

# install deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt

# copy app code
COPY backend/ .
RUN python -m py_compile app_pipeline.py lambda_handler.py

# Lambda container entrypoint
ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["lambda_handler.lambda_handler"]