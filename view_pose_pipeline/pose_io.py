from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from .io_utils import parse_pose_file


def load_pose_sequences(result_dirs: Dict[str, Path], frame_ids: Iterable[int]):
    pose_sequences = {}
    for cam_id, result_dir in result_dirs.items():
        frames = {}
        for frame_id in frame_ids:
            pose_path = result_dir / f"per_frame_refined_poses_{frame_id:06d}.json"
            pose = parse_pose_file(pose_path)
            if pose is not None:
                frames[frame_id] = pose
        pose_sequences[cam_id] = frames
    return pose_sequences
