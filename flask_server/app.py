import os
import io
import time
import uuid
import base64
import logging
import requests
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

DETECTOR_URL = os.getenv("DETECTOR_URL", "http://localhost:5001")
GROUPING_URL = os.getenv("GROUPING_URL", "http://localhost:5002")
OUTPUT_DIR   = Path(os.getenv("OUTPUT_DIR", "./visualizations"))
MAX_SIZE     = 20 * 1024 * 1024
ALLOWED      = {"png", "jpg", "jpeg", "webp", "bmp"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="../templates", static_folder="../static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_SIZE


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def to_b64(data):
    return base64.b64encode(data).decode("utf-8")

def call_detector(img_b64):
    r = requests.post(f"{DETECTOR_URL}/detect", json={"image_b64": img_b64}, timeout=60)
    r.raise_for_status()
    return r.json()

def call_grouping(img_b64, detections):
    r = requests.post(f"{GROUPING_URL}/group", json={"image_b64": img_b64, "detections": detections}, timeout=60)
    r.raise_for_status()
    return r.json()


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/visualizations/<path:filename>")
def serve_vis(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route("/analyze", methods=["POST"])
def analyze():
    t0  = time.perf_counter()
    rid = str(uuid.uuid4())[:8]

    try:
        if request.content_type and "multipart" in request.content_type:
            if "image" not in request.files:
                return jsonify({"error": "no image in request"}), 400
            f = request.files["image"]
            if not allowed(f.filename):
                return jsonify({"error": "file type not supported"}), 415
            img_bytes = f.read()
            fname = secure_filename(f.filename)
        else:
            body = request.get_json(force=True)
            if not body or "image_b64" not in body:
                return jsonify({"error": "need image_b64 in body"}), 400
            img_bytes = base64.b64decode(body["image_b64"])
            fname = body.get("filename", "upload.jpg")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    img_b64 = to_b64(img_bytes)
    log.info("[%s] received %s (%d bytes)", rid, fname, len(img_bytes))

    try:
        det_out = call_detector(img_b64)
    except requests.RequestException as e:
        return jsonify({"error": "detector is down", "detail": str(e)}), 503

    detections = det_out.get("detections", [])
    log.info("[%s] %d products detected", rid, len(detections))

    try:
        grp_out = call_grouping(img_b64, detections)
    except requests.RequestException as e:
        return jsonify({"error": "grouping service is down", "detail": str(e)}), 503

    final_detections = grp_out.get("detections", detections)
    groups           = grp_out.get("groups", {})
    vis_b64          = grp_out.get("visualization_b64", "")

    vis_file = ""
    if vis_b64:
        vis_file = f"{rid}_result.jpg"
        with open(OUTPUT_DIR / vis_file, "wb") as fh:
            fh.write(base64.b64decode(vis_b64))

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    log.info("[%s] finished in %s ms", rid, elapsed)

    return jsonify({
        "request_id":         rid,
        "image_filename":     fname,
        "total_detections":   len(final_detections),
        "total_groups":       len(groups),
        "processing_time_ms": elapsed,
        "visualization_path": f"/visualizations/{vis_file}" if vis_file else "",
        "visualization_b64":  vis_b64,
        "detections":         final_detections,
        "groups":             groups,
    }), 200

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
