import io
import os
import base64
import logging
import colorsys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# try loading mobilenet for better embeddings, fall back to colour histograms if not available
try:
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as T

    _net = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
    _net.classifier = torch.nn.Identity()
    _net.eval()
    _tfm = T.Compose([
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    USE_CNN = True
    log.info("mobilenetv2 loaded")
except Exception as e:
    USE_CNN = False
    log.warning("mobilenetv2 not available (%s), using colour histograms only", e)


def hsv_hist(crop):
    arr  = np.array(crop.convert("HSV"), dtype=np.float32)
    h    = np.histogram(arr[:, :, 0], bins=16, range=(0, 255))[0]
    s    = np.histogram(arr[:, :, 1], bins=8,  range=(0, 255))[0]
    v    = np.histogram(arr[:, :, 2], bins=8,  range=(0, 255))[0]
    feat = np.concatenate([h, s, v]).astype(np.float32)
    return feat / (feat.sum() + 1e-8)


def cnn_embed(crop):
    import torch
    x = _tfm(crop).unsqueeze(0)
    with torch.no_grad():
        out = _net.features(x)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        return out.view(-1).numpy().astype(np.float32)


def get_features(image, detections):
    feats = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        crop = image.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else image.crop((0, 0, 32, 32))

        hist = hsv_hist(crop)
        feat = np.concatenate([hist * 2.0, cnn_embed(crop)]) if USE_CNN else hist
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


def cluster(features):
    n = len(features)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    X = normalize(features)
    if X.shape[1] > 32:
        pca = PCA(n_components=min(32, X.shape[0] - 1), random_state=42)
        X   = normalize(pca.fit_transform(X))

    from sklearn.metrics import pairwise_distances
    dists  = pairwise_distances(X, metric="cosine")
    np.fill_diagonal(dists, np.nan)
    eps    = float(np.clip(np.nanmean(dists) / 2.5, 0.10, 0.55))

    return DBSCAN(eps=eps, min_samples=2, metric="cosine").fit_predict(X)


def make_palette(n):
    colors = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def draw(image, detections, color_map):
    vis  = image.copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = tuple(color_map.get(det.get("group_id", ""), (180, 180, 180)))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = det.get("group_id", "?")
        tb    = draw.textbbox((x1, y1 - 16), label, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((x1, y1 - 16), label, fill=(255, 255, 255), font=font)
    return vis


def to_b64(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/group", methods=["POST"])
def group_route():
    body       = request.get_json(force=True)
    detections = body.get("detections", [])
    img_b64    = body.get("image_b64", "")

    try:
        image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"bad image: {e}"}), 400

    t0 = time.perf_counter()

    if not detections:
        return jsonify({"detections": [], "groups": {}, "visualization_b64": to_b64(image)}), 200

    features = get_features(image, detections)
    labels   = cluster(features)

    unique   = sorted(set(l for l in labels if l >= 0))
    palette  = make_palette(len(unique))

    color_map = {}
    groups    = {}
    enriched  = []

    for det, lbl in zip(detections, labels):
        gid   = f"grp_{lbl:03d}" if lbl >= 0 else f"grp_s_{det['detection_id']:04d}"
        color = palette[unique.index(lbl)] if lbl >= 0 else (180, 180, 180)
        color_map[gid] = color

        if gid not in groups:
            groups[gid] = {"count": 0, "color": list(color), "detection_ids": []}
        groups[gid]["count"] += 1
        groups[gid]["detection_ids"].append(det["detection_id"])
        enriched.append({**det, "group_id": gid, "group_color": list(color)})

    vis_b64 = to_b64(draw(image, enriched, color_map))
    ms      = round((time.perf_counter() - t0) * 1000, 2)
    log.info("grouped %d detections into %d groups in %s ms", len(detections), len(groups), ms)

    return jsonify({"detections": enriched, "groups": groups, "visualization_b64": vis_b64}), 200


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
