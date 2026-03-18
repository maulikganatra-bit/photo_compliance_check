"""
Experiment runner for license plate detection benchmarking.
"""
from typing import List, Dict, Any, Optional
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from detectors.yolo_detector import YOLODetector
from detectors.llm_openai import OpenAIGPT4oJudge
from detectors.llm_gemini import GeminiJudge
from detectors.llm_claude import ClaudeJudge
from utils.image_utils import list_images
from utils.logging_utils import save_csv, save_json
from evaluation.metrics import compute_agreement
from config import Config


class ExperimentRunner:
    def __init__(self, config: Config):
        self.config = config
        self.yolo = YOLODetector(
            model_path=config.YOLO_MODEL_PATH,
            conf_threshold=config.YOLO_CONF_THRESHOLD,
            save_annotated=config.YOLO_SAVE_ANNOTATED,
            output_dir=config.YOLO_ANNOTATED_DIR
        )
        self.llms = []
        if config.USE_OPENAI and config.OPENAI_API_KEY:
            self.llms.append(("openai_result", OpenAIGPT4oJudge(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)))
        if config.USE_GEMINI and config.GEMINI_API_KEY:
            self.llms.append(("gemini_result", GeminiJudge(api_key=config.GEMINI_API_KEY, model=config.GEMINI_MODEL)))
        if config.USE_CLAUDE and config.CLAUDE_API_KEY:
            self.llms.append(("claude_result", ClaudeJudge(api_key=config.CLAUDE_API_KEY, model=config.CLAUDE_MODEL)))
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def _process_image(self, img_path: str) -> Dict[str, Any]:
        """Process a single image with YOLO and LLMs (sync mode)."""
        row = {"image_name": os.path.basename(img_path)}

        # YOLO detection
        yolo_result = self.yolo.detect(img_path)
        row["yolo_result"] = yolo_result

        # Parallel LLM processing with timeout
        if self.llms:
            with ThreadPoolExecutor(max_workers=len(self.llms)) as executor:
                futures = {
                    executor.submit(llm.analyze, img_path): llm_name
                    for llm_name, llm in self.llms
                }
                for future in as_completed(futures, timeout=120):
                    llm_name = futures[future]
                    try:
                        result = future.result()
                        row[llm_name] = result
                    except Exception as e:
                        row[llm_name] = {"license_plate_detected": None, "confidence": 0, "reasoning": f"Error: {str(e)[:50]}"}

        return row

    async def _analyze_llm_task(self, llm_name: str, llm, img_path: str, img_index: int) -> tuple:
        """Run a single async LLM analysis. Returns (img_index, llm_name, result)."""
        try:
            result = await asyncio.wait_for(llm.analyze_async(img_path), timeout=120)
            return (img_index, llm_name, result)
        except Exception as e:
            return (img_index, llm_name, {"license_plate_detected": None, "confidence": 0, "reasoning": f"Error: {str(e)[:50]}"})

    async def run_async(self, image_dir: Optional[str] = None):
        """Async experiment runner — YOLO sequential, LLM calls concurrent via async/await."""
        image_dir = image_dir or self.config.IMAGE_DIR
        images = list_images(image_dir)

        if not images:
            print("No images found in directory.")
            return

        print(f"Processing {len(images)} images (async mode)...")

        # Step 1: Run YOLO sequentially (CPU-bound, not thread-safe)
        print("Running YOLO detection...")
        results = []
        for i, img_path in enumerate(images, 1):
            print(f"  [YOLO {i}/{len(images)}] {os.path.basename(img_path)}...", end=" ", flush=True)
            row = {"image_name": os.path.basename(img_path)}
            yolo_result = self.yolo.detect(img_path)
            row["yolo_result"] = yolo_result
            results.append(row)
            print("✓")

        # Step 2: Fire all LLM calls concurrently
        if self.llms:
            tasks = []
            for i, img_path in enumerate(images):
                for llm_name, llm in self.llms:
                    tasks.append(self._analyze_llm_task(llm_name, llm, img_path, i))

            total_calls = len(tasks)
            llm_names = [name for name, _ in self.llms]
            print(f"Running {total_calls} LLM calls concurrently ({', '.join(llm_names)})...")

            completed = await asyncio.gather(*tasks)

            for img_index, llm_name, result in completed:
                results[img_index][llm_name] = result

            print(f"  All {total_calls} LLM calls complete. ✓")

        # Step 3: Compute metrics
        yolo_results = [r["yolo_result"] for r in results]
        llm_results = [[] for _ in self.llms]
        for row in results:
            for idx, (llm_name, _) in enumerate(self.llms):
                if llm_name in row:
                    llm_results[idx].append(row[llm_name])

        metrics = compute_agreement(yolo_results, llm_results)
        for idx in metrics["disagreement_cases"]:
            results[idx]["disagreement"] = True

        # Step 4: Save results
        save_csv(results, os.path.join(self.config.OUTPUT_DIR, "results.csv"))
        save_json(results, os.path.join(self.config.OUTPUT_DIR, "results.json"))
        save_json(metrics, os.path.join(self.config.OUTPUT_DIR, "metrics.json"))
        disagreements = [results[idx] for idx in metrics["disagreement_cases"]]
        save_json(disagreements, os.path.join(self.config.OUTPUT_DIR, "disagreements.json"))

        # Step 5: Print summary
        self._print_summary(images, metrics)

    def run(self, image_dir: Optional[str] = None):
        """Sync experiment runner — sequential image processing."""
        image_dir = image_dir or self.config.IMAGE_DIR
        images = list_images(image_dir)

        if not images:
            print("No images found in directory.")
            return

        results = []
        yolo_results = []
        llm_results = [[] for _ in self.llms]

        print(f"Processing {len(images)} images...")
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {os.path.basename(img_path)}...", end=" ", flush=True)
            row = self._process_image(img_path)
            results.append(row)

            yolo_results.append(row["yolo_result"])
            for idx, (llm_name, _) in enumerate(self.llms):
                if llm_name in row:
                    llm_results[idx].append(row[llm_name])

            print("✓")

        metrics = compute_agreement(yolo_results, llm_results)
        for idx in metrics["disagreement_cases"]:
            results[idx]["disagreement"] = True

        # Save results
        save_csv(results, os.path.join(self.config.OUTPUT_DIR, "results.csv"))
        save_json(results, os.path.join(self.config.OUTPUT_DIR, "results.json"))
        save_json(metrics, os.path.join(self.config.OUTPUT_DIR, "metrics.json"))
        disagreements = [results[idx] for idx in metrics["disagreement_cases"]]
        save_json(disagreements, os.path.join(self.config.OUTPUT_DIR, "disagreements.json"))

        self._print_summary(images, metrics)

    def _print_summary(self, images, metrics):
        print("\n" + "="*60)
        print("Experiment complete.")
        print(f"Total images: {len(images)}")
        print(f"YOLO flagged: {metrics['yolo_flagged_pct']*100:.1f}%")
        for i, (llm_name, _) in enumerate(self.llms):
            print(f"{llm_name} flagged: {metrics['llm_flagged_pct'][i]*100:.1f}% | Agreement: {metrics['agreement_rates'][i]*100:.1f}%")
        print(f"Disagreement cases: {len(metrics['disagreement_cases'])}")
        print("="*60)
