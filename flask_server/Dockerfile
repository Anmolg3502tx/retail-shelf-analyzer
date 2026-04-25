FROM python:3.11-slim

WORKDIR /app

COPY flask_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_server/app.py .
COPY templates/ templates/
COPY static/ static/

ENV PYTHONUNBUFFERED=1

EXPOSE 5000
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "--timeout=120", "app:app"]
