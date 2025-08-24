FROM python:3.11-slim

# system deps: ffmpeg for audio extraction
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

# python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt

# app code
COPY . .

# run via AWS Lambda Runtime Interface Client
CMD ["python", "-m", "awslambdaric", "lambda_handler.lambda_handler"]
