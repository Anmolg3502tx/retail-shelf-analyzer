import argparse
import base64
import json
import sys
import time
from pathlib import Path
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--url",    default="http://localhost:5000")
    parser.add_argument("--save",   default="")
    parser.add_argument("--json",   action="store_true")
    args = parser.parse_args()

    img = Path(args.image)
    if not img.exists():
        print(f"file not found: {img}"); sys.exit(1)

    print(f"sending {img.name} to {args.url} ...")

    t0 = time.perf_counter()
    with open(img, "rb") as f:
        resp = requests.post(f"{args.url}/analyze", files={"image": (img.name, f, "image/jpeg")}, timeout=120)
    rtt = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        print(f"error {resp.status_code}: {resp.text}"); sys.exit(1)

    data = resp.json()
    print(f"\nrequest_id      : {data['request_id']}")
    print(f"detections      : {data['total_detections']}")
    print(f"groups          : {data['total_groups']}")
    print(f"pipeline time   : {data['processing_time_ms']} ms")
    print(f"round trip      : {rtt:.1f} ms")

    print("\ngroups breakdown:")
    for gid, info in (data.get("groups") or {}).items():
        r, g, b = info["color"]
        print(f"  {gid:20s} {info['count']} products  rgb({r},{g},{b})")

    if data.get("visualization_b64"):
        out = args.save or f"result_{data['request_id']}.jpg"
        with open(out, "wb") as f:
            f.write(base64.b64decode(data["visualization_b64"]))
        print(f"\nannotated image saved to {out}")

    if args.json:
        clean = {k: v for k, v in data.items() if k != "visualization_b64"}
        print(json.dumps(clean, indent=2))


if __name__ == "__main__":
    main()
