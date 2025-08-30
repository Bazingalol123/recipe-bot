# backend_api/web_api.py
import json
import os

def handler(event, context):
    route = event.get("requestContext", {}).get("http", {}).get("path", "")
    method = event.get("requestContext", {}).get("http", {}).get("method", "")

    # JWT claims are injected by API Gateway HTTP API (JWT authorizer)
    claims = (event.get("requestContext", {})
                    .get("authorizer", {})
                    .get("jwt", {})
                    .get("claims", {}))

    if route == "/books" and method == "GET":
        user = claims.get("email") or claims.get("cognito:username") or "guest"
        return _ok([{"book_id": "demo", "title": f"{user}'s Recipe Book"}])

    if route == "/books" and method == "POST":
        data = json.loads(event.get("body") or "{}")
        return _ok({"created": True, "title": data.get("title", "Untitled")})

    return _err(404, "Not Found")

def _ok(data):  # helpers
    return {"statusCode": 200, "headers": {"content-type": "application/json"},
            "body": json.dumps(data, ensure_ascii=False)}

def _err(code, msg):
    return {"statusCode": code, "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": msg}, ensure_ascii=False)}
# Note: To test locally, set environment variable LAMBDA_ARN to your Lambda function ARN
