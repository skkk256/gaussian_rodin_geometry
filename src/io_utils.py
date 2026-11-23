import hashlib
import json
import os
from pathlib import Path

import tempfile

import cv2
import numpy as np

OUTPUT_FILES = ('rodin.glb', 'transform.json', 'fal_refine.png', 'gaussian.png', 'gaussian_filter.png', 'gaussian_filter.ply', 'rodin.png', 'timing.json')


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    """PNG file -> RGB uint8 [H,W,3]."""
    rgb = cv2.imread(str(path))
    assert rgb is not None, path
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def png_bytes(rgb: np.ndarray) -> bytes:
    """RGB uint8 [H,W,3] -> encoded PNG bytes."""
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()


def write_rgb(path: Path, rgb: np.ndarray) -> None:
    """RGB uint8 [H,W,3] -> PNG file."""
    path.write_bytes(png_bytes(rgb))


def join_views(views: np.ndarray) -> np.ndarray:
    """RGB uint8 [4,1024,1024,3] -> RGB uint8 [2048,2048,3], front/right/back/left in row-major order."""
    assert views.shape == (4, 1024, 1024, 3) and views.dtype == np.uint8
    return np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0)


def split_views(rgb: np.ndarray) -> list[np.ndarray]:
    """RGB uint8 [2048,2048,3] -> four RGB uint8 [1024,1024,3] views in row-major order."""
    assert rgb.shape == (2048, 2048, 3) and rgb.dtype == np.uint8
    return [rgb[:1024, :1024], rgb[:1024, 1024:], rgb[1024:, :1024], rgb[1024:, 1024:]]


def validate_glb(path: Path) -> None:
    body = path.read_bytes()
    assert len(body) >= 20 and body[:4] == b'glTF', path
    assert int.from_bytes(body[4:8], 'little') == 2
    assert int.from_bytes(body[8:12], 'little') == len(body)
