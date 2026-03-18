"""
Dataset experiment runner — processes all images from photo_compliance_data.csv
using OpenAI GPT-4o vision (URL-based, no image download needed).

Features:
- Passes image URLs directly to OpenAI (no local download)
- Async concurrent API calls with configurable concurrency limit
- Incremental CSV writes (resume-safe)
- Skips already-processed URLs on re-run
"""
import asyncio
import csv
import json
import os
import time

from dotenv import load_dotenv

load_dotenv(override=True)

from detectors.llm_openai import OpenAIGPT4oJudge

# -- Configuration -------------------------------------------------------------
DATASET_CSV    = "./dataset/photo_compliance_data.csv"
OUTPUT_CSV     = "./results/dataset_openai_results_v2.csv"
OPENAI_MODEL   = "gpt-4o"
MAX_CONCURRENT = 10   # concurrent OpenAI API calls
OUTPUT_FIELDNAMES = [
    "url", "photo_id", "mlsnum", "display_order",
    "license_plate_visible", "confidence", "detected_vehicle_count",
    "vehicles_with_visible_plate", "reasoning", "error"
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


async def process_row(row: dict, judge: OpenAIGPT4oJudge, semaphore: asyncio.Semaphore) -> dict:
    result = {
        "url":          row["url"],
        "photo_id":     row["photo_id"],
        "mlsnum":       row["mlsnum"],
        "display_order": row["display_order"],
        "license_plate_visible":       None,
        "confidence":                  None,
        "detected_vehicle_count":      None,
        "vehicles_with_visible_plate": None,
        "reasoning":                   None,
        "error":                       None,
    }
    async with semaphore:
        detection = await judge.analyze_from_url_async(row["url"])

    if detection.get("license_plate_visible") is None:
        result["error"] = detection.get("reasoning", "unknown error")
    else:
        result["license_plate_visible"]       = detection.get("license_plate_visible")
        result["confidence"]                  = detection.get("confidence")
        result["detected_vehicle_count"]      = detection.get("detected_vehicle_count")
        result["vehicles_with_visible_plate"] = detection.get("vehicles_with_visible_plate")
        result["reasoning"]                   = detection.get("reasoning", "")
    return result


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

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

    judge = OpenAIGPT4oJudge(api_key=api_key, model=OPENAI_MODEL)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    out_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDNAMES)
    if write_header:
        writer.writeheader()

    processed = 0
    detected  = 0
    errors    = 0
    start_time = time.time()

    print(f"Starting with max {MAX_CONCURRENT} concurrent OpenAI calls...\n")

    tasks = [process_row(row, judge, semaphore) for row in pending]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        processed += 1

        if result["error"]:
            errors += 1
        elif result["license_plate_visible"]:
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
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
