"""
Temporal pose filtering and smoothing.

Two strategies:
- HardRejectFilter  : keep previous pose when jump exceeds thresholds (original behaviour).
- EMAFilter         : exponential moving average — smoother, no hard cuts.
"""
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .transforms import rotation_angle_deg, split_transform, transform_matrix


# ---------------------------------------------------------------------------
# Hard-reject filter (original behaviour, preserved for compatibility)
# ---------------------------------------------------------------------------

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

    prev_R, prev_t = split_transform(previous_pose_world)
    curr_R, curr_t = split_transform(current_pose_world)
    dt = float(np.linalg.norm(curr_t - prev_t))
    dr = rotation_angle_deg(prev_R, curr_R)

    if dt <= translation_thresh_m and dr <= rotation_thresh_deg:
        return current_pose_world, None

    print(
        f"[pose-filter] frame {frame_id:06d} rejected, "
        f"dt={dt:.4f}m dr={dr:.2f}deg — using previous pose."
    )
    return (
        previous_pose_world.copy(),
        f"replaced by prev | dt={dt:.3f}m | dr={dr:.1f}deg",
    )


# ---------------------------------------------------------------------------
# EMA filter — SHOW3D-inspired temporal smoothing
# ---------------------------------------------------------------------------

class EMAFilter:
    """
    Exponential moving average filter for 6-DoF pose.

    Translation: t_out = alpha * t_new + (1-alpha) * t_prev
    Rotation   : SLERP(R_prev, R_new, alpha)

    alpha=1.0 → no smoothing (pass-through).
    alpha=0.5 → heavy smoothing.
    Recommended: alpha_t=0.7, alpha_r=0.6 for real-time tracking.
    """

    def __init__(self, alpha_t: float = 0.7, alpha_r: float = 0.6):
        self.alpha_t = alpha_t
        self.alpha_r = alpha_r
        self._pose: Optional[np.ndarray] = None

    def reset(self):
        self._pose = None

    def update(self, pose: np.ndarray) -> np.ndarray:
        if self._pose is None:
            self._pose = pose.copy()
            return self._pose

        prev_R, prev_t = split_transform(self._pose)
        curr_R, curr_t = split_transform(pose)

        # Translation EMA
        t_out = self.alpha_t * curr_t + (1.0 - self.alpha_t) * prev_t

        # Rotation SLERP
        rots = Rotation.from_matrix(np.stack([prev_R, curr_R]))
        slerp = Slerp([0.0, 1.0], rots)
        R_out = slerp(self.alpha_r).as_matrix()

        self._pose = transform_matrix(R_out, t_out)
        return self._pose

    @property
    def last(self) -> Optional[np.ndarray]:
        return self._pose
