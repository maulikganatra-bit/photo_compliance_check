"""
Logging and result saving utilities for experiments.
"""
from typing import List, Dict, Any
import csv
import json
import os

def save_csv(results: List[Dict[str, Any]], csv_path: str):
    if not results:
        return
    keys = sorted(results[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

def save_json(data: Any, json_path: str):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
