FROM public.ecr.aws/lambda/python:3.11

COPY functions/api/requirements.txt /var/task/requirements.txt
RUN python -m pip install --no-cache-dir -r /var/task/requirements.txt -t /var/task
COPY functions/api/backend_api/ /var/task/backend_api/
CMD ["backend_api.web_api.handler"]