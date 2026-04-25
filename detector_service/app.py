import os
import io
import base64
import logging
import time

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

CONF  = float(os.getenv("MODEL_CONF", 0.25))
IOU   = float(os.getenv("MODEL_IOU",  0.45))
MODEL = os.getenv("MODEL_PATH", "yolov8n.pt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

log.info("loading model %s", MODEL)
model = YOLO(MODEL)
model.fuse()
log.info("model ready")


def decode_image(b64_str):
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.route("/detect", methods=["POST"])
def detect():
    body = request.get_json(force=True)
    if not body or "image_b64" not in body:
        return jsonify({"error": "missing image_b64"}), 400

    try:
        img = decode_image(body["image_b64"])
    except Exception as e:
        return jsonify({"error": f"bad image: {e}"}), 400

    w, h = img.size
    t0 = time.perf_counter()

    results = model.predict(source=img, conf=CONF, iou=IOU, verbose=False)

    ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info("inference done in %s ms", ms)

    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i, (box, conf, cls) in enumerate(zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())):
            x1, y1, x2, y2 = [round(v) for v in box]
            detections.append({
                "detection_id": i,
                "bbox":         [x1, y1, x2, y2],
                "confidence":   round(float(conf), 4),
                "class_id":     int(cls),
                "class_name":   model.names[int(cls)],
            })

    log.info("%d detections", len(detections))
    return jsonify({
        "detections":   detections,
        "image_width":  w,
        "image_height": h,
        "inference_ms": ms,
    }), 200


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
