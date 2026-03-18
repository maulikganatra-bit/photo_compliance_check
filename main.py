"""
CLI entry point for license plate detection experiment framework.
"""
import argparse
import asyncio
from config import Config
from evaluation.experiment_runner import ExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(description="License Plate Detection Experiment Framework")
    parser.add_argument('--image_dir', type=str, help='Directory of images to process')
    parser.add_argument('--use_openai', type=bool, default=None, help='Use OpenAI GPT-4o')
    parser.add_argument('--use_gemini', type=bool, default=None, help='Use Gemini 2.0 Flash')
    parser.add_argument('--use_claude', type=bool, default=None, help='Use Claude Sonnet 4')
    parser.add_argument('--yolo_model', type=str, default=None, help='Path to YOLO weights')
    parser.add_argument('--yolo_conf', type=float, default=None, help='YOLO confidence threshold')
    parser.add_argument('--openai_model', type=str, default=None, help='OpenAI model version')
    parser.add_argument('--gemini_model', type=str, default=None, help='Gemini model version')
    parser.add_argument('--claude_model', type=str, default=None, help='Claude model version')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    if args.image_dir:
        config.IMAGE_DIR = args.image_dir
    if args.use_openai is not None:
        config.USE_OPENAI = args.use_openai
    if args.use_gemini is not None:
        config.USE_GEMINI = args.use_gemini
    if args.use_claude is not None:
        config.USE_CLAUDE = args.use_claude
    if args.yolo_model:
        config.YOLO_MODEL_PATH = args.yolo_model
    if args.yolo_conf is not None:
        config.YOLO_CONF_THRESHOLD = args.yolo_conf
    if args.openai_model:
        config.OPENAI_MODEL = args.openai_model
    if args.gemini_model:
        config.GEMINI_MODEL = args.gemini_model
    if args.claude_model:
        config.CLAUDE_MODEL = args.claude_model
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir

    runner = ExperimentRunner(config)
    if config.MULTIPROCESSING:
        asyncio.run(runner.run_async())
    else:
        runner.run()

if __name__ == "__main__":
    main()
