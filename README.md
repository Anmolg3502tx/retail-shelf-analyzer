# Retail Shelf Analyzer

**Author: Anmol Gupta**

A microservices-based AI pipeline that detects products on retail shelves and groups them by brand using computer vision.

## Architecture

Three independent Flask services communicate over HTTP:

| Service | Port | Role |
|---|---|---|
| `flask_server` | 5000 | Gateway + Web UI |
| `detector_service` | 5001 | YOLOv8-nano object detection |
| `grouping_service` | 5002 | MobileNetV2 + DBSCAN brand grouping |

## Features

 **Product Detection** – YOLOv8-nano detects all products on a shelf
**Brand Grouping** – DBSCAN clusters products by visual similarity (colour + CNN embeddings)
**Web UI** – Drag-and-drop image upload with annotated output and JSON response
 **Docker support** – One command to spin up all services
**CPU-only** – No GPU required; runs on any laptop

## Quick Start

### Docker
```bash
docker compose up --build
# open http://localhost:5000
```

### Local Python (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r flask_server/requirements.txt -r detector_service/requirements.txt -r grouping_service/requirements.txt
# Start each in a separate terminal:
.\.venv\Scripts\python detector_service/app.py
.\.venv\Scripts\python grouping_service/app.py
.\.venv\Scripts\python flask_server/app.py
```

Then open **http://localhost:5000**

## API

```
POST /analyze   multipart/form-data  field: image
POST /analyze   application/json     body: { "image_b64": "..." }
```

### Response
```json
{
  "request_id": "a1b2c3d4",
  "total_detections": 47,
  "total_groups": 6,
  "processing_time_ms": 810.4,
  "detections": [...],
  "groups": {...}
}
```

## Tech Stack

- **Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Embeddings**: MobileNetV2 (torchvision)
- **Clustering**: DBSCAN (scikit-learn)
- **Backend**: Flask + Flask-CORS
- **Frontend**: Vanilla HTML/CSS/JS

## Project Structure

```
retail_shelf_analyzer/
├── flask_server/       ← Gateway + Web UI (port 5000)
├── detector_service/   ← YOLOv8 detection (port 5001)
├── grouping_service/   ← Brand grouping (port 5002)
├── templates/          ← HTML frontend
├── docs/               ← Solution write-up
├── docker-compose.yml
├── run_local.sh
└── test_pipeline.py
```

## Testing

```bash
python test_pipeline.py --image shelf.jpg --save result.jpg
```

Health checks:
```bash
curl http://localhost:5000/health
curl http://localhost:5001/health
curl http://localhost:5002/health
```
