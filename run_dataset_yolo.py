"""
Dataset experiment runner — processes all images from photo_compliance_data.csv
using the local YOLO license plate detector (license_plate_detector.pt).

Features:
- Downloads images from URLs on the fly (no pre-download needed)
- Concurrent downloads + inference via ThreadPoolExecutor
- Saves annotated images with bounding boxes drawn
- Incremental CSV writes (resume-safe)
- Skips already-processed URLs on re-run
"""
import csv
import json
import os
import time
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# -- Configuration -------------------------------------------------------------
DATASET_CSV       = "./dataset/photo_compliance_data.csv"
OUTPUT_CSV        = "./results/dataset_yolo_results.csv"
ANNOTATED_DIR     = "./results/annotated_yolo"
MODEL_PATH        = "./license_plate_detector.pt"
CONF_THRESHOLD    = 0.25   # minimum confidence to count a detection
MAX_WORKERS       = 8      # concurrent download+inference threads
DOWNLOAD_TIMEOUT  = 15     # seconds per image download

OUTPUT_FIELDNAMES = [
    "url", "photo_id", "mlsnum", "display_order",
    "license_plate_detected", "num_detections", "confidence",
    "bounding_boxes", "annotated_image_path", "error"
]
# ------------------------------------------------------------------------------


def load_done_urls(output_path: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("url"):
                done.add(row["url"])
    return done


def download_image(url: str) -> np.ndarray:
    """Download image from URL and return as BGR numpy array."""
    resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from URL")
    return img


def safe_filename(photo_id: str, url: str) -> str:
    """Generate a safe filename for the annotated image."""
    if photo_id:
        return f"{photo_id}.jpg"
    # fallback: last path segment of URL
    segment = url.rstrip("/").split("/")[-1].split("?")[0]
    return f"{segment[:60]}.jpg" if segment else f"{abs(hash(url))}.jpg"


def process_row(row: dict, model: YOLO) -> dict:
    result = {
        "url":                  row["url"],
        "photo_id":             row["photo_id"],
        "mlsnum":               row["mlsnum"],
        "display_order":        row["display_order"],
        "license_plate_detected": None,
        "num_detections":       None,
        "confidence":           None,
        "bounding_boxes":       None,
        "annotated_image_path": None,
        "error":                None,
    }

    try:
        img = download_image(row["url"])

        yolo_results = model(img, verbose=False)[0]

        detections = [
            d for d in yolo_results.boxes.data.cpu().numpy()
            if d[4] >= CONF_THRESHOLD
        ]

        bounding_boxes = []
        for d in detections:
            x1, y1, x2, y2, conf, cls = d
            bounding_boxes.append({
                "box": [round(float(x1), 1), round(float(y1), 1),
                        round(float(x2), 1), round(float(y2), 1)],
                "confidence": round(float(conf), 4)
            })

        result["license_plate_detected"] = len(bounding_boxes) > 0
        result["num_detections"]         = len(bounding_boxes)
        result["confidence"]             = round(
            max((b["confidence"] for b in bounding_boxes), default=0.0), 4
        )
        result["bounding_boxes"] = json.dumps(bounding_boxes)

        # Save annotated image
        annotated = yolo_results.plot()  # BGR numpy array with boxes drawn
        fname = safe_filename(row["photo_id"], row["url"])
        out_path = os.path.join(ANNOTATED_DIR, fname)
        cv2.imwrite(out_path, annotated)
        result["annotated_image_path"] = out_path

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(ANNOTATED_DIR, exist_ok=True)

    # Load dataset
    rows = []
    with open(DATASET_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "url":           r["Url"],
                "photo_id":      r.get("PhotoID", ""),
                "mlsnum":        r.get("Mlsnum", ""),
                "display_order": r.get("DisplayOrder", ""),
            })

    total = len(rows)
    print(f"Dataset: {total} rows loaded from {DATASET_CSV}")

    done_urls = load_done_urls(OUTPUT_CSV)
    pending = [r for r in rows if r["url"] not in done_urls]
    print(f"Already processed: {len(done_urls)} | Remaining: {len(pending)}")

    if not pending:
        print("All rows already processed. Done.")
        return

    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print(f"Model loaded. Starting with {MAX_WORKERS} workers...\n")

    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    out_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDNAMES)
    if write_header:
        writer.writeheader()

    processed = 0
    detected  = 0
    errors    = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row, model): row for row in pending}

        for future in as_completed(futures):
            result = future.result()
            processed += 1

            if result["error"]:
                errors += 1
            elif result["license_plate_detected"]:
                detected += 1

            writer.writerow(result)
            out_file.flush()

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
    print(f"  Annotated images saved  : {ANNOTATED_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
