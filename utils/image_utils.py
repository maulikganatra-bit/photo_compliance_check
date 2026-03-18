"""
Image utility functions for license plate detection experiments.
"""
from typing import List
import os

def list_images(directory: str, extensions: List[str] = [".jpg", ".jpeg", ".png"]) -> List[str]:
    """
    List all image files in a directory with given extensions.
    """
    files = []
    for fname in os.listdir(directory):
        if any(fname.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(directory, fname))
    return sorted(files)
