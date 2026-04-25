FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY detector_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights at build time
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY detector_service/app.py .

ENV PYTHONUNBUFFERED=1

EXPOSE 5001
CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:5001", "--timeout=120", "app:app"]
