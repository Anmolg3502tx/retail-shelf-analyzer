# Retail Shelf Analyzer – Solution Write-up

**Author: Anmol Gupta**

## How it works (quick summary)

There are three services talking to each other:

1. **Flask gateway (port 5000)** – takes the image from the browser, calls the other two services one after the other, saves the annotated output image, and sends back JSON.
2. **Detector (port 5001)** – runs YOLOv8-nano on the image and returns bounding boxes.
3. **Grouping (port 5002)** – takes those boxes, crops each product out of the image, builds a feature vector (colour histogram + MobileNetV2 embedding), clusters them with DBSCAN, draws colour-coded boxes, and returns the annotated image.

All three communicate over HTTP using base64-encoded images in the JSON body — no shared file paths anywhere.

---

## Why these specific choices

**YOLOv8-nano for detection**

I went with YOLOv8n because it's tiny (~6 MB weights) and runs fine on CPU without needing a GPU. Latency on a normal laptop is around 200–400 ms which is acceptable. It's pretrained on COCO which covers bottles, boxes, cans etc. — basically the shapes you'd see on a retail shelf. If this needed to go to production, I'd fine-tune it on SKU-110K or the Grocery Products dataset; the code doesn't need to change at all, just swap the `.pt` file.

**DBSCAN for grouping**

The main reason I didn't use K-Means is that you don't know how many brands are on a shelf ahead of time. DBSCAN figures out the number of clusters on its own and doesn't force lone products into a wrong group — it just marks them as singletons. I'm using cosine distance on normalised embeddings which works well here since we care about the shape of the feature vector, not its magnitude.

For the features themselves: HSV colour histograms are cheap and capture packaging colour really well (which is usually the strongest brand signal). MobileNetV2 adds texture and shape on top of that. I weight the histogram slightly higher since it's more directly relevant.

**Why microservices**

Keeps the detector and grouper independently scalable. If inference is the bottleneck you can spin up more detector containers without touching the gateway. Each service is stateless so you can put a load balancer in front of any of them.

---

## JSON format

**Input to `/analyze`** (either works):
```
POST /analyze  multipart/form-data   field: image
POST /analyze  application/json      body: { "image_b64": "..." }
```

**Output from `/analyze`**:
```json
{
  "request_id": "a1b2c3d4",
  "image_filename": "shelf.jpg",
  "total_detections": 47,
  "total_groups": 6,
  "processing_time_ms": 810.4,
  "visualization_path": "/visualizations/a1b2c3d4_result.jpg",
  "visualization_b64": "<base64 jpeg>",
  "detections": [
    {
      "detection_id": 0,
      "bbox": [34, 120, 98, 210],
      "confidence": 0.71,
      "class_id": 39,
      "class_name": "bottle",
      "group_id": "grp_001",
      "group_color": [255, 85, 85]
    }
  ],
  "groups": {
    "grp_001": {
      "count": 12,
      "color": [255, 85, 85],
      "detection_ids": [0, 3, 7, 11]
    }
  }
}
```

---

## Setup

### Docker (easiest)
```bash
docker compose up --build
# open http://localhost:5000
```

Model weights download automatically inside the container on first build.

### Local Python (no Docker)
```bash
bash run_local.sh
# open http://localhost:5000
```

This creates a virtualenv, installs everything, and starts all three services.

### Test from terminal
```bash
python test_pipeline.py --image shelf.jpg --save result.jpg
```

### Health checks
```bash
curl http://localhost:5000/health
curl http://localhost:5001/health
curl http://localhost:5002/health
```

---

## Other approaches I considered

**For detection:**
- RT-DETR gives better accuracy but is much heavier and doesn't run well on CPU.
- Grounding DINO would let you use text prompts ("find all shampoo bottles") but needs a GPU.
- A fine-tuned model on SKU-110K would be ideal for production — I'd go that route if time allowed.

**For grouping:**
- K-Means: simple but you need to specify K, which isn't realistic here.
- CLIP embeddings: semantically very powerful but requires GPU or an API call.
- Template matching: would work perfectly if you had a product database, but that's a lot of setup.

**For scaling in production:**
- Put Nginx in front of multiple gateway instances.
- Use a job queue (Celery + Redis) so the API returns immediately with a job ID and the client polls for results.
- TorchServe or Triton for batched GPU inference on the detector.
- Replace base64 transport with pre-signed S3 URLs to cut serialisation overhead.

---

## Files

```
retail_shelf_analyzer/
├── docker-compose.yml
├── run_local.sh
├── test_pipeline.py
├── flask_server/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
├── detector_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
├── grouping_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
├── templates/
│   └── index.html
├── visualizations/        ← output images saved here
└── docs/
    └── WRITEUP.md
```
