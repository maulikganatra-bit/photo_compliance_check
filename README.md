# PhotoCompliance - License Plate Detection Benchmark

A benchmarking framework that compares **YOLOv8** (computer vision) against **multimodal LLMs** (GPT-4o, Gemini 2.0 Flash, Claude Sonnet 4) for detecting vehicle license plates in property listing images.

Built for **MLS (Multiple Listing Service) compliance** workflows, where real estate listing photos must not contain visible license plates.

## Project Structure

```
PhotoCompliance/
├── main.py                          # CLI entry point
├── config.py                        # Configuration (models, API keys, thresholds)
├── requirements.txt                 # Python dependencies
├── license_plate_detector.pt        # YOLOv8 model trained on license plates
├── .env                             # API keys (not committed)
├── detectors/
│   ├── yolo_detector.py             # YOLOv8 license plate detector
│   ├── llm_openai.py                # OpenAI GPT-4o vision judge
│   ├── llm_gemini.py                # Google Gemini 2.0 Flash judge
│   └── llm_claude.py                # Anthropic Claude Sonnet 4 judge
├── evaluation/
│   ├── experiment_runner.py          # Orchestrates the full pipeline
│   └── metrics.py                   # Agreement/disagreement metrics
├── utils/
│   ├── image_utils.py               # Image file listing utilities
│   └── logging_utils.py             # CSV/JSON result saving
├── images/                          # Input images (add your own)
└── results/                         # Output directory (auto-created)
    ├── results.csv                  # Flat results for spreadsheet analysis
    ├── results.json                 # Full structured results
    ├── metrics.json                 # Agreement statistics
    └── disagreements.json           # Cases where detectors disagree
```

## How It Works

```
Input Images (JPG/PNG)
        |
        v
 [ExperimentRunner]
        |
   +---------+--------------------+
   |                              |
[YOLOv8]              [LLM Judges (parallel)]
 license_plate            +-- OpenAI GPT-4o
 detector.pt              +-- Gemini 2.0 Flash
   |                      +-- Claude Sonnet 4
   +---------+--------------------+
             |
             v
    [Agreement Metrics]
             |
   +---------+---------+-----------+
   |         |         |           |
results.  results.  metrics.  disagreements.
  csv       json      json       json
```

1. Each image is run through **YOLOv8** for object-level license plate detection (bounding boxes + confidence).
2. The same image is sent to **enabled LLM judges** in parallel, each returning a structured JSON verdict (`license_plate_detected`, `confidence`, `reasoning`).
3. **Agreement metrics** are computed between YOLO and each LLM, including majority voting and disagreement tracking.

## Setup

### Prerequisites

- Python 3.10+
- API keys for the LLMs you want to use (at least one)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/PhotoCompliance.git
cd PhotoCompliance
```

### 2. Create a Virtual Environment

```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** If you encounter a `LongPathsEnabled` error during PyTorch installation, enable Windows Long Path support:
> 1. Open Registry Editor (`regedit`) as Administrator
> 2. Navigate to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
> 3. Set `LongPathsEnabled` to `1`
> 4. Restart your terminal and retry

### 4. Download the YOLO License Plate Model

Download `license_plate_detector.pt` from [Google Drive](https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view?usp=sharing) and place it in the project root.

Or using `gdown`:

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw" -O license_plate_detector.pt
```

This is a YOLOv8 model fine-tuned on the [Roboflow License Plate Recognition dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4). It detects a single class: `license_plate`.

### 5. Configure API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
```

You only need keys for the LLMs you plan to enable.

### 6. Add Images

Place your property listing images (`.jpg`, `.jpeg`, `.png`) in the `./images` directory.

## Usage

### Run with Default Settings

```bash
python main.py
```

This uses the defaults from `config.py`:
- YOLO: `license_plate_detector.pt` with 0.5 confidence threshold
- OpenAI GPT-4o: enabled
- Claude Sonnet 4: enabled
- Gemini: disabled

### CLI Options

Override any setting via command-line arguments:

```bash
# Use a different image directory
python main.py --image_dir ./my_photos

# Set YOLO confidence threshold
python main.py --yolo_conf 0.3

# Enable/disable specific LLMs
python main.py --use_openai True --use_gemini True --use_claude False

# Use a different YOLO model
python main.py --yolo_model yolov8n.pt

# Custom output directory
python main.py --output_dir ./my_results

# Override LLM model versions
python main.py --openai_model gpt-4o --claude_model claude-sonnet-4-20250514
```

### Toggle LLMs in Config

Edit `config.py` to change defaults:

```python
USE_OPENAI: bool = True
USE_GEMINI: bool = False   # Set to True to enable
USE_CLAUDE: bool = True
```

## Output

Results are saved to `./results/` (configurable):

| File | Description |
|------|-------------|
| `results.csv` | Flat format for spreadsheet analysis |
| `results.json` | Full structured results with all detector outputs |
| `metrics.json` | Agreement rates, flagging percentages, majority votes |
| `disagreements.json` | Only the images where detectors disagreed |

### Sample Console Output

```
Processing 5 images...
[1/5] property_001.png...
0: 448x640 1 license_plate, 116.7ms
✓
[2/5] property_002.png...
0: 448x640 2 license_plates, 69.1ms
✓

============================================================
Experiment complete.
Total images: 5
YOLO flagged: 100.0%
openai_result flagged: 100.0% | Agreement: 100.0%
claude_result flagged: 100.0% | Agreement: 100.0%
Disagreement cases: 0
============================================================
```

### Sample Result (results.json)

```json
{
  "image_name": "property_001.png",
  "yolo_result": {
    "license_plate_detected": true,
    "confidence": 0.715,
    "num_detections": 1,
    "bounding_boxes": [
      {
        "box": [795.95, 549.33, 915.95, 605.96],
        "confidence": 0.715
      }
    ]
  },
  "openai_result": {
    "license_plate_detected": true,
    "confidence": 95,
    "reasoning": "A clearly visible alphanumeric plate is attached to the rear of the vehicle."
  },
  "claude_result": {
    "license_plate_detected": true,
    "confidence": 95,
    "reasoning": "Clear rear license plate visible on sedan showing alphanumeric characters."
  }
}
```

## Configuration Reference

All defaults are set in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `YOLO_MODEL_PATH` | `license_plate_detector.pt` | Path to YOLO weights |
| `YOLO_CONF_THRESHOLD` | `0.5` | Minimum confidence for detections |
| `YOLO_SAVE_ANNOTATED` | `False` | Save images with bounding box overlays |
| `YOLO_ANNOTATED_DIR` | `./annotated` | Output dir for annotated images |
| `USE_OPENAI` | `True` | Enable OpenAI GPT-4o judge |
| `USE_GEMINI` | `False` | Enable Google Gemini judge |
| `USE_CLAUDE` | `True` | Enable Anthropic Claude judge |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model version |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model version |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model version |
| `IMAGE_DIR` | `./images` | Input image directory |
| `OUTPUT_DIR` | `./results` | Output directory for results |

## Notes

- **LLM calls run in parallel** using `ThreadPoolExecutor` with a 120-second timeout per call.
- **Retry logic**: Each LLM judge retries up to 3 times with a 2-second delay on API failures.
- **YOLO runs locally** and does not require an internet connection or API key.
- The `google-generativeai` package used by the Gemini detector is deprecated. Consider migrating to `google-genai` for long-term support.

## Credits

The YOLO license plate model (`license_plate_detector.pt`) was trained on the [Roboflow License Plate Recognition dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) by [Muhammad Zeerak Khan](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8).
