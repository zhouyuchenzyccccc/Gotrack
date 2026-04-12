from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .depth_utils import DepthCache, project_points_to_depth_image
from .transforms import average_transforms, split_transform


@dataclass
class FusionResult:
    pose_world: Optional[np.ndarray]
    selected_cam_ids: List[str]
    evaluations: Dict[str, Dict[str, object]]
    confidence: float
    status: Optional[str] = None
    mode: str = "multiview"


def ransac_view_consensus(
    candidate_poses_world: Dict[str, Tuple[np.ndarray, float]],
    thresh_m: float,
    min_views: int,
) -> Dict[str, Tuple[np.ndarray, float]]:
    if len(candidate_poses_world) <= 2:
        return candidate_poses_world

    cam_ids = list(candidate_poses_world.keys())
    translations = np.asarray(
        [candidate_poses_world[cam_id][0][:3, 3] for cam_id in cam_ids],
        dtype=np.float64,
    )
    scores = np.asarray(
        [max(candidate_poses_world[cam_id][1], 1e-6) for cam_id in cam_ids],
        dtype=np.float64,
    )

    best_mask = np.ones(len(cam_ids), dtype=bool)
    best_count = 0
    best_score_sum = -1.0
    for idx in range(len(cam_ids)):
        distances = np.linalg.norm(translations - translations[idx][None, :], axis=1)
        inlier_mask = distances <= thresh_m
        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_score_sum = float(np.sum(scores[inlier_mask]))
        if inlier_count > best_count or (
            inlier_count == best_count and inlier_score_sum > best_score_sum
        ):
            best_mask = inlier_mask
            best_count = inlier_count
            best_score_sum = inlier_score_sum

    if best_count < min_views:
        return candidate_poses_world
    return {
        cam_ids[i]: candidate_poses_world[cam_ids[i]]
        for i in range(len(cam_ids))
        if best_mask[i]
    }


def evaluate_pose_multiview(
    pose_world: np.ndarray,
    camera_ids: List[str],
    depth_cache: DepthCache,
    camera_params: Dict[str, Dict],
    extrinsics: Dict[str, Dict],
    mesh_points_m: np.ndarray,
    max_depth_m: float,
    inlier_thresh_m: float,
    source_score: float = 1.0,
) -> Dict[str, object]:
    r_world_obj, t_world_obj = split_transform(pose_world)
    object_points_world = (r_world_obj @ mesh_points_m.T).T + t_world_obj[None, :]

    total_valid = 0
    total_inliers = 0
    residual_sum = 0.0
    valid_views = 0
    per_view = {}
    for cam_id in camera_ids:
        depth_m = depth_cache.get_depth_m(cam_id)
        if depth_m is None:
            per_view[cam_id] = {"valid": 0, "inliers": 0, "mae_mm": float("inf")}
            continue
        h, w = depth_m.shape[:2]
        uv, points_depth = project_points_to_depth_image(
            object_points_world, camera_params, extrinsics, cam_id
        )
        x = np.rint(uv[:, 0]).astype(np.int32)
        y = np.rint(uv[:, 1]).astype(np.int32)
        z_pred = points_depth[:, 2]
        inside = (
            (x >= 0)
            & (x < w)
            & (y >= 0)
            & (y < h)
            & (z_pred > 0.05)
            & (z_pred < max_depth_m)
        )
        if not np.any(inside):
            per_view[cam_id] = {"valid": 0, "inliers": 0, "mae_mm": float("inf")}
            continue

        obs = depth_m[y[inside], x[inside]]
        valid_obs = obs > 0.05
        if not np.any(valid_obs):
            per_view[cam_id] = {"valid": 0, "inliers": 0, "mae_mm": float("inf")}
            continue

        pred = z_pred[inside][valid_obs]
        obs = obs[valid_obs]
        residual_m = np.abs(obs - pred)
        inliers = residual_m < inlier_thresh_m
        valid_count = int(len(residual_m))
        inlier_count = int(np.count_nonzero(inliers))
        mae_mm = float(np.mean(residual_m) * 1000.0)
        total_valid += valid_count
        total_inliers += inlier_count
        residual_sum += float(np.sum(residual_m))
        valid_views += 1
        per_view[cam_id] = {
            "valid": valid_count,
            "inliers": inlier_count,
            "mae_mm": mae_mm,
        }

    coverage = float(total_valid) / float(max(1, len(camera_ids) * len(mesh_points_m)))
    inlier_ratio = float(total_inliers) / float(max(1, total_valid))
    mean_residual_m = residual_sum / float(max(1, total_valid))
    consistency = float(inlier_ratio * max(coverage, 1e-6) ** 0.5)
    confidence = float(max(source_score, 1e-6) * consistency)
    total_cost = mean_residual_m + 0.04 * (1.0 - inlier_ratio) + 0.02 * (1.0 - coverage)
    return {
        "cost": float(total_cost),
        "coverage": coverage,
        "inlier_ratio": inlier_ratio,
        "consistency": consistency,
        "confidence": confidence,
        "valid_views": valid_views,
        "source_score": float(source_score),
        "per_view": per_view,
    }


def try_track_from_previous_pose(
    previous_pose_world: Optional[np.ndarray],
    previous_confidence: float,
    camera_ids: List[str],
    depth_cache: DepthCache,
    camera_params: Dict[str, Dict],
    extrinsics: Dict[str, Dict],
    mesh_points_m: np.ndarray,
    max_depth_m: float,
    inlier_thresh_m: float,
    confidence_thresh: float,
    consistency_thresh: float,
) -> Optional[FusionResult]:
    if previous_pose_world is None or previous_confidence < confidence_thresh:
        return None
    evaluation = evaluate_pose_multiview(
        pose_world=previous_pose_world,
        camera_ids=camera_ids,
        depth_cache=depth_cache,
        camera_params=camera_params,
        extrinsics=extrinsics,
        mesh_points_m=mesh_points_m,
        max_depth_m=max_depth_m,
        inlier_thresh_m=inlier_thresh_m,
        source_score=previous_confidence,
    )
    if (
        float(evaluation["consistency"]) < consistency_thresh
        or int(evaluation["valid_views"]) == 0
    ):
        return None
    return FusionResult(
        pose_world=previous_pose_world.copy(),
        selected_cam_ids=[],
        evaluations={"prev_track": evaluation},
        confidence=float(evaluation["confidence"]),
        status=(
            "tracked from prev"
            f" | conf={float(evaluation['confidence']):.3f}"
            f" | consistency={float(evaluation['consistency']):.3f}"
        ),
        mode="track_prev",
    )


def select_and_refine_multiview_pose(
    candidate_poses_world: Dict[str, Tuple[np.ndarray, float]],
    camera_ids: List[str],
    depth_cache: DepthCache,
    camera_params: Dict[str, Dict],
    extrinsics: Dict[str, Dict],
    mesh_points_m: np.ndarray,
    max_depth_m: float,
    inlier_thresh_m: float,
    view_consensus_thresh_m: float,
    view_consensus_min_views: int,
    candidate_cost_margin: float,
) -> FusionResult:
    if not candidate_poses_world:
        return FusionResult(None, [], {}, 0.0, mode="multiview")

    consensus_candidates = ransac_view_consensus(
        candidate_poses_world,
        thresh_m=view_consensus_thresh_m,
        min_views=view_consensus_min_views,
    )
    evaluations: Dict[str, Dict[str, object]] = {}
    for cam_id, (pose_world, score) in consensus_candidates.items():
        evaluations[cam_id] = evaluate_pose_multiview(
            pose_world=pose_world,
            camera_ids=camera_ids,
            depth_cache=depth_cache,
            camera_params=camera_params,
            extrinsics=extrinsics,
            mesh_points_m=mesh_points_m,
            max_depth_m=max_depth_m,
            inlier_thresh_m=inlier_thresh_m,
            source_score=float(score),
        )

    ranked = sorted(
        evaluations.items(),
        key=lambda item: (
            item[1]["cost"],
            -item[1]["consistency"],
            -item[1]["source_score"],
        ),
    )
    best_cam_id = ranked[0][0]
    best_cost = float(ranked[0][1]["cost"])
    selected_cam_ids = []
    selected_transforms = []
    selected_weights = []
    for cam_id, eval_info in ranked:
        if float(eval_info["cost"]) <= best_cost + candidate_cost_margin:
            selected_cam_ids.append(cam_id)
            selected_transforms.append(consensus_candidates[cam_id][0])
            selected_weights.append(max(1e-6, float(eval_info["confidence"])))

    if not selected_transforms:
        selected_cam_ids = [best_cam_id]
        selected_transforms = [consensus_candidates[best_cam_id][0]]
        selected_weights = [1.0]

    selected_weights_np = np.asarray(selected_weights, dtype=np.float64)
    fused_pose = average_transforms(selected_transforms, selected_weights_np)
    fused_eval = evaluate_pose_multiview(
        pose_world=fused_pose,
        camera_ids=camera_ids,
        depth_cache=depth_cache,
        camera_params=camera_params,
        extrinsics=extrinsics,
        mesh_points_m=mesh_points_m,
        max_depth_m=max_depth_m,
        inlier_thresh_m=inlier_thresh_m,
        source_score=float(np.mean(selected_weights_np)),
    )
    evaluations["fused"] = fused_eval
    return FusionResult(
        pose_world=fused_pose,
        selected_cam_ids=selected_cam_ids,
        evaluations=evaluations,
        confidence=float(fused_eval["confidence"]),
        status=(
            "fused"
            f" | conf={float(fused_eval['confidence']):.3f}"
            f" | consistency={float(fused_eval['consistency']):.3f}"
        ),
        mode="multiview",
    )
