from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def read_image(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def load_json(path: Path):
    return json.loads(path.read_text())


def collect_available_frames(results_dir: Path) -> List[int]:
    frame_ids = set()
    for pattern in [
        "processed_detections_*.png",
        "vis_*_foundPose.png",
        "vis_*_goTrack.png",
    ]:
        for path in results_dir.glob(pattern):
            for part in path.stem.split("_"):
                if part.isdigit():
                    frame_ids.add(int(part))
                    break
    return sorted(frame_ids)


def build_frame_paths(results_dir: Path, frame_id: int) -> Dict[str, Path]:
    return {
        "detection": results_dir / f"processed_detections_{frame_id:06d}.png",
        "coarse": results_dir / f"vis_{frame_id:06d}_foundPose.png",
        "refined": results_dir / f"vis_{frame_id:06d}_goTrack.png",
    }


def copy_rgb_to_png(src: Path, dst: Path):
    if dst.exists():
        return
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), image)


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    dst.write_bytes(src.read_bytes())


def shutil_rmtree(path: Path):
    for child in sorted(path.glob("**/*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    if path.exists():
        path.rmdir()


def collect_frame_ids_from_raw(raw_data_dir: Path, camera_ids: List[str]) -> List[int]:
    frame_sets = []
    for cam_id in camera_ids:
        rgb_dir = raw_data_dir / cam_id / "RGB"
        frame_sets.append({int(path.stem) for path in rgb_dir.glob("*.jpg")})
    common = set.intersection(*frame_sets) if frame_sets else set()
    return sorted(common)


def parse_pose_file(path: Path):
    if not path.exists():
        return None
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None
    row = rows[0]
    return {
        "R": np.fromstring(row["R"], sep=" ", dtype=np.float64).reshape(3, 3),
        "t_m": np.fromstring(row["t"], sep=" ", dtype=np.float64) / 1000.0,
        "score": float(row["score"]),
    }
