"""
YOLOv8 License Plate Detector
"""
from typing import Dict, Any, Optional
import os
from ultralytics import YOLO
import cv2

class YOLODetector:
    """
    YOLOv8-based license plate detector.
    Supports Ultralytics model names (e.g., 'yolov8n.pt') or local paths.
    """
    def __init__(self, model_path: str, conf_threshold: float = 0.5, save_annotated: bool = False, output_dir: Optional[str] = None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.save_annotated = save_annotated
        self.output_dir = output_dir
        if save_annotated and output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def detect_from_array(self, img) -> Dict[str, Any]:
        """Run detection on a numpy image array (BGR or RGB)."""
        results = self.model(img)[0]
        detections = [d for d in results.boxes.data.cpu().numpy() if d[4] >= self.conf_threshold]
        bounding_boxes = []
        for d in detections:
            x1, y1, x2, y2, conf, cls = d
            bounding_boxes.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf)
            })
        license_plate_detected = len(bounding_boxes) > 0
        confidence = float(max([b["confidence"] for b in bounding_boxes], default=0.0))
        return {
            "license_plate_detected": license_plate_detected,
            "confidence": confidence,
            "num_detections": len(bounding_boxes),
            "bounding_boxes": bounding_boxes
        }

    def detect(self, image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        results = self.model(img)[0]
        detections = [d for d in results.boxes.data.cpu().numpy() if d[4] >= self.conf_threshold]
        bounding_boxes = []
        for d in detections:
            x1, y1, x2, y2, conf, cls = d
            bounding_boxes.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf)
            })
        license_plate_detected = len(bounding_boxes) > 0
        confidence = float(max([b["confidence"] for b in bounding_boxes], default=0.0))
        result = {
            "license_plate_detected": license_plate_detected,
            "confidence": confidence,
            "num_detections": len(bounding_boxes),
            "bounding_boxes": bounding_boxes
        }
        if self.save_annotated and self.output_dir:
            annotated = results.plot()
            out_path = os.path.join(self.output_dir, os.path.basename(image_path))
            cv2.imwrite(out_path, annotated)
        return result
