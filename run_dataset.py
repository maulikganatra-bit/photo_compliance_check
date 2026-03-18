"""
Dataset experiment runner — processes all images from photo_compliance_data.csv
via URL download and YOLO license plate detection.

Features:
- Downloads images in memory (no disk writes)
- Concurrent processing with ThreadPoolExecutor
- Incremental CSV writes (resume-safe)
- Skips already-processed URLs on re-run
"""
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

from detectors.yolo_detector import YOLODetector

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_CSV = "./dataset/photo_compliance_data.csv"
OUTPUT_CSV  = "./results/dataset_results.csv"
MODEL_PATH  = "./license_plate_detector.pt"
CONF_THRESHOLD = 0.5
NUM_WORKERS    = 8          # concurrent download + inference threads
REQUEST_TIMEOUT = 15        # seconds per image download
OUTPUT_FIELDNAMES = [
    "url", "photo_id", "mlsnum", "display_order",
    "license_plate_detected", "confidence", "num_detections",
    "bounding_boxes", "error"
]
# ──────────────────────────────────────────────────────────────────────────────


def load_done_urls(output_path: str) -> set:
    """Return set of URLs already present in the output CSV."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("url"):
                done.add(row["url"])
    return done


def download_image(url: str, timeout: int = REQUEST_TIMEOUT):
    """Download image from URL and return as BGR numpy array."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    img_pil = Image.open(BytesIO(resp.content)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_bgr


def process_row(row: dict, detector: YOLODetector, lock: threading.Lock) -> dict:
    """Download image and run YOLO. Returns result dict."""
    url          = row["url"]
    photo_id     = row["photo_id"]
    mlsnum       = row["mlsnum"]
    display_order = row["display_order"]

    result = {
        "url": url,
        "photo_id": photo_id,
        "mlsnum": mlsnum,
        "display_order": display_order,
        "license_plate_detected": None,
        "confidence": None,
        "num_detections": None,
        "bounding_boxes": None,
        "error": None,
    }

    try:
        img = download_image(url)
    except Exception as e:
        result["error"] = f"download_error: {str(e)[:120]}"
        return result

    try:
        with lock:
            detection = detector.detect_from_array(img)
        result["license_plate_detected"] = detection["license_plate_detected"]
        result["confidence"]             = detection["confidence"]
        result["num_detections"]         = detection["num_detections"]
        result["bounding_boxes"]         = json.dumps(detection["bounding_boxes"])
    except Exception as e:
        result["error"] = f"yolo_error: {str(e)[:120]}"

    return result


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Load dataset
    rows = []
    with open(DATASET_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "url":           r["Url"],
                "photo_id":      r.get("PhotoID", ""),
                "mlsnum":        r.get("Mlsnum", ""),
                "display_order": r.get("DisplayOrder", ""),
            })

    total = len(rows)
    print(f"Dataset: {total} rows loaded from {DATASET_CSV}")

    # Resume: skip already-processed URLs
    done_urls = load_done_urls(OUTPUT_CSV)
    pending = [r for r in rows if r["url"] not in done_urls]
    print(f"Already processed: {len(done_urls)} | Remaining: {len(pending)}")

    if not pending:
        print("All rows already processed. Done.")
        return

    # Load YOLO model (once, shared across threads)
    print(f"Loading YOLO model: {MODEL_PATH}")
    detector = YOLODetector(
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        save_annotated=False,
    )
    yolo_lock = threading.Lock()  # Serialize YOLO inference (thread safety)

    # Open output CSV (append mode to support resume)
    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    out_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDNAMES)
    if write_header:
        writer.writeheader()

    write_lock = threading.Lock()

    # Progress counters
    processed = 0
    errors     = 0
    detected   = 0
    start_time = time.time()

    print(f"Starting processing with {NUM_WORKERS} workers...\n")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_row, row, detector, yolo_lock): row
            for row in pending
        }

        for future in as_completed(futures):
            result = future.result()
            processed += 1

            if result["error"]:
                errors += 1
            elif result["license_plate_detected"]:
                detected += 1

            with write_lock:
                writer.writerow(result)
                out_file.flush()

            # Progress update every 50 images
            if processed % 50 == 0 or processed == len(pending):
                elapsed = time.time() - start_time
                rate    = processed / elapsed if elapsed > 0 else 0
                eta_s   = (len(pending) - processed) / rate if rate > 0 else 0
                print(
                    f"  [{processed}/{len(pending)}] "
                    f"detected={detected} errors={errors} "
                    f"rate={rate:.1f} img/s  ETA={eta_s/60:.1f}min"
                )

    out_file.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done. Processed {processed} images in {elapsed/60:.1f} min.")
    print(f"  License plates detected : {detected} ({detected/processed*100:.1f}%)")
    print(f"  Errors                  : {errors} ({errors/processed*100:.1f}%)")
    print(f"  Results saved to        : {OUTPUT_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
