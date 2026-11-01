"""Reject mismatched scene/result visualizations rather than draw misleading overlays."""
import hashlib
import numpy as np

def validate_result(image: np.ndarray,result: dict) -> None:
    if image.shape[:2]!=(result['height'],result['width']):raise ValueError('Result/image dimensions differ')
    if hashlib.sha256(image.tobytes()).hexdigest()!=result['image_sha256']:raise ValueError('Result belongs to different RGB pixels')
