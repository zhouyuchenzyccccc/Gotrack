"""
Multi-view pose fusion inspired by SHOW3D (CVPR 2023).

Improvements over naive weighted average:
1. Multi-view depth consistency: each candidate pose is scored against ALL
   cameras' depth maps, not just the source camera.
2. RANSAC-based view consensus: find the largest set of cameras whose
   translation estimates agree within a threshold before fusing.
3. Consistency-weighted fusion: fuse using depth-consistency scores as weights.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .depth_utils import DepthCache, project_mesh_to_depth
from .transforms import (
    split_transform,
    transform_matrix,
    weighted_average_quaternions,
)


# ---------------------------------------------------------------------------
# Per-view depth consistency scoring
# ---------------------------------------------------------------------------

def _score_pose_vs_depth(
    pose_world: np.ndarray,
    depth: np.ndarray,
    camera_params: Dict,
    extrinsics: Dict,
    cam_id: str,
    mesh_points_m: np.ndarray,
    max_depth_m: float,
    inlier_thresh_m: float,
) -> float:
    """Inlier ratio of mesh reprojection against one depth image (0..1)."""
    R, t = split_transform(pose_world)
    obj_pts = (R @ mesh_points_m.T).T + t[None, :]
    uv, pts_d = project_mesh_to_depth(obj_pts, camera_params, extrinsics, cam_id)

    h, w = depth.shape[:2]
    depth_m = depth.astype(np.float32) / 1000.0
    xi = np.rint(uv[:, 0]).astype(np.int32)
    yi = np.rint(uv[:, 1]).astype(np.int32)
    z_pred = pts_d[:, 2]

    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (z_pred > 0.05) & (z_pred < max_depth_m)
    if not np.any(inside):
        return 0.0
    obs = depth_m[yi[inside], xi[inside]]
    valid = obs > 0.05
    if not np.any(valid):
        return 0.0
    residual = np.abs(obs[valid] - z_pred[inside][valid])
    return float(np.mean(residual < inlier_thresh_m))


def compute_multiview_consistency(
    pose_world: np.ndarray,
    frame_id: int,
    camera_ids: List[str],
    raw_data_dir: Path,
    camera_params: Dict,
    extrinsics: Dict,
    mesh_points_m: np.ndarray,
    max_depth_m: float,
    depth_cache: Optional[DepthCache] = None,
    inlier_thresh_m: float = 0.02,
) -> float:
    """Mean inlier ratio of a pose across ALL cameras' depth maps."""
    scores = []
    for cam_id in camera_ids:
        depth_path = raw_data_dir / cam_id / "Depth" / f"{frame_id:05d}.png"
        depth = (
            depth_cache.get(cam_id, frame_id, depth_path)
            if depth_cache is not None
            else cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        )
        if depth is None:
            continue
        s = _score_pose_vs_depth(
            pose_world, depth, camera_params, extrinsics, cam_id,
            mesh_points_m, max_depth_m, inlier_thresh_m,
        )
        scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# RANSAC-based view consensus
# ---------------------------------------------------------------------------

def ransac_view_consensus(
    candidate_poses_world: Dict[str, Tuple[np.ndarray, float]],
    translation_thresh_m: float = 0.08,
) -> List[str]:
    """Return the largest subset of cameras whose translations agree."""
    cam_ids = list(candidate_poses_world.keys())
    if len(cam_ids) <= 1:
        return cam_ids

    translations = np.array([candidate_poses_world[c][0][:3, 3] for c in cam_ids])
    best: List[str] = [cam_ids[0]]
    for i in range(len(cam_ids)):
        dists = np.linalg.norm(translations - translations[i][None, :], axis=1)
        inliers = [cam_ids[j] for j, d in enumerate(dists) if d <= translation_thresh_m]
        if len(inliers) > len(best):
            best = inliers
    return best


# ---------------------------------------------------------------------------
# Main fusion entry point
# ---------------------------------------------------------------------------

def show3d_fuse_poses(
    candidate_poses_world: Dict[str, Tuple[np.ndarray, float]],
    frame_id: int,
    camera_ids: List[str],
    raw_data_dir: Path,
    camera_params: Dict,
    extrinsics: Dict,
    mesh_points_m: np.ndarray,
    max_depth_m: float,
    depth_cache: Optional[DepthCache] = None,
    ransac_thresh_m: float = 0.08,
    inlier_thresh_m: float = 0.02,
) -> Tuple[Optional[np.ndarray], List[str], Dict[str, float]]:
    """
    SHOW3D-inspired multi-view pose fusion.

    Steps:
      1. RANSAC consensus to remove outlier cameras.
      2. Score each remaining candidate against ALL depth maps.
      3. Fuse using consistency-weighted average (translation + SLERP rotation).

    Returns (fused_pose_world, selected_cam_ids, consistency_scores).
    """
    if not candidate_poses_world:
        return None, [], {}

    # Step 1: RANSAC consensus
    consensus_ids = ransac_view_consensus(candidate_poses_world, ransac_thresh_m)
    consensus_poses = {c: candidate_poses_world[c] for c in consensus_ids}

    # Step 2: multi-view depth consistency score for each consensus camera
    consistency: Dict[str, float] = {}
    for cam_id, (pose_world, _) in consensus_poses.items():
        consistency[cam_id] = compute_multiview_consistency(
            pose_world, frame_id, camera_ids,
            raw_data_dir, camera_params, extrinsics,
            mesh_points_m, max_depth_m, depth_cache, inlier_thresh_m,
        )

    # Step 3: weighted fusion
    # Weight = detection_score * depth_consistency
    weights = np.array(
        [max(1e-6, consensus_poses[c][1]) * max(1e-6, consistency[c]) for c in consensus_ids],
        dtype=np.float64,
    )
    transforms = [consensus_poses[c][0] for c in consensus_ids]

    fused_t = np.average(
        np.array([tf[:3, 3] for tf in transforms], dtype=np.float64),
        axis=0, weights=weights,
    )
    quats = Rotation.from_matrix(
        np.array([tf[:3, :3] for tf in transforms], dtype=np.float64)
    ).as_quat()
    fused_q = weighted_average_quaternions(quats, weights)
    fused_R = Rotation.from_quat(fused_q).as_matrix()

    return transform_matrix(fused_R, fused_t), consensus_ids, consistency
