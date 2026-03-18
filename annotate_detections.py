"""
Draws bounding boxes on detected images using stored results from dataset_results.csv.
No YOLO re-run — uses saved bounding_boxes coordinates directly.
Saves annotated images to ./annotated/dataset/
"""
import csv
import json
import os
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
from PIL import Image

INPUT_CSV   = "./results/dataset_results.csv"
OUTPUT_DIR  = "./annotated/dataset"
NUM_WORKERS = 8
REQUEST_TIMEOUT = 15
BOX_COLOR   = (0, 0, 255)   # Red in BGR
BOX_THICKNESS = 3
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.8
FONT_THICKNESS = 2


def draw_boxes(img: np.ndarray, bounding_boxes: list) -> np.ndarray:
    annotated = img.copy()
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = [int(v) for v in bbox["box"]]
        conf = bbox["confidence"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"plate {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        # Background rect for text
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), FONT, FONT_SCALE,
                    (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
    return annotated


def safe_filename(photo_id: str, url: str) -> str:
    """Build a safe output filename from photo_id."""
    pid = str(photo_id).replace("-", "").strip() or "unknown"
    return f"{pid}.jpg"


def process(row: dict, output_dir: str) -> str:
    url      = row["url"]
    photo_id = row["photo_id"]
    bboxes   = json.loads(row["bounding_boxes"])

    # Download
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    img_pil = Image.open(BytesIO(resp.content)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Draw boxes
    annotated = draw_boxes(img_bgr, bboxes)

    # Save
    fname = safe_filename(photo_id, url)
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, annotated)
    return out_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load detected rows only
    detected_rows = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("license_plate_detected") == "True":
                detected_rows.append(row)

    total = len(detected_rows)
    print(f"Found {total} detected images to annotate -> {OUTPUT_DIR}\n")

    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process, row, OUTPUT_DIR): row for row in detected_rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                out_path = future.result()
                done += 1
                print(f"  [{done}/{total}] saved: {os.path.basename(out_path)}")
            except Exception as e:
                errors += 1
                print(f"  ERROR photo_id={row['photo_id']}: {e}")

    print(f"\nDone. {done} annotated, {errors} errors -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
