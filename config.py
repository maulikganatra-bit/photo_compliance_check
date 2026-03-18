"""
Experiment configuration.
"""
from typing import Optional
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(override=True)

class Config:
    # YOLO
    YOLO_MODEL_PATH: str = "license_plate_detector.pt"
    YOLO_CONF_THRESHOLD: float = 0.5
    YOLO_SAVE_ANNOTATED: bool = True
    YOLO_ANNOTATED_DIR: Optional[str] = "./annotated"

    # LLMs
    USE_OPENAI: bool = True
    USE_GEMINI: bool = False
    USE_CLAUDE: bool = True
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    CLAUDE_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_MODEL: str = "gpt-4o"
    GEMINI_MODEL: str = "gemini-2.0-flash"
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"

    # Experiment
    IMAGE_DIR: str = "./images"
    OUTPUT_DIR: str = "./results"
    MULTIPROCESSING: bool = True
    NUM_WORKERS: int = 4
