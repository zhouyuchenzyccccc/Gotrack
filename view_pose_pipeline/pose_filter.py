from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .transforms import rotation_angle_deg, split_transform


def compute_pose_jump(
    previous_pose_world: np.ndarray, current_pose_world: np.ndarray
) -> Tuple[float, float]:
    prev_r, prev_t = split_transform(previous_pose_world)
    curr_r, curr_t = split_transform(current_pose_world)
    translation_jump = float(np.linalg.norm(curr_t - prev_t))
    rotation_jump = rotation_angle_deg(prev_r, curr_r)
    return translation_jump, rotation_jump


def filter_large_pose_jump(
    frame_id: int,
    current_pose_world: Optional[np.ndarray],
    previous_pose_world: Optional[np.ndarray],
    enabled: bool,
    translation_thresh_m: float,
    rotation_thresh_deg: float,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if current_pose_world is None or previous_pose_world is None or not enabled:
        return current_pose_world, None

    translation_jump, rotation_jump = compute_pose_jump(
        previous_pose_world, current_pose_world
    )
    if (
        translation_jump <= translation_thresh_m
        and rotation_jump <= rotation_thresh_deg
    ):
        return current_pose_world, None

    print(
        "[pose-filter] frame "
        f"{frame_id:06d} rejected, "
        f"translation_jump={translation_jump:.4f} m, "
        f"rotation_jump={rotation_jump:.2f} deg. "
        "Using previous accepted pose."
    )
    return (
        previous_pose_world.copy(),
        f"replaced by prev | dt={translation_jump:.3f}m | dr={rotation_jump:.1f}deg",
    )

