FROM python:3.12-slim

WORKDIR /app

# Prevent Python from creating .pyc files and make logs appear immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy inference server and pre-trained weights
COPY app/     ./app/
COPY weights/ ./weights/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
