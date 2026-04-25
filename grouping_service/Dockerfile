FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY grouping_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MobileNetV2 weights
RUN python -c "import torchvision.models as m; m.mobilenet_v2(weights=m.MobileNet_V2_Weights.DEFAULT)"

COPY grouping_service/app.py .

ENV PYTHONUNBUFFERED=1

EXPOSE 5002
CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:5002", "--timeout=120", "app:app"]
