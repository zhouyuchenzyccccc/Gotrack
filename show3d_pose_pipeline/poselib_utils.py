from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from view_pose_pipeline.transforms import split_transform, transform_matrix, world_from_rgb_extrinsic

try:
    import poselib
except Exception:
    poselib = None


@dataclass
class GeneralizedCorrespondenceSet:
    points2d: List[np.ndarray]
    points3d: List[np.ndarray]
    camera_ext: List[object]
    camera_dicts: List[dict]


def require_poselib():
    if poselib is None:
        raise ImportError(
            "PoseLib is required for SHOW3D-style generalized PnP. "
            "Install it with `pip install poselib` in the target environment."
        )


def make_poselib_camera_dict(intrinsic: Dict[str, float]) -> dict:
    return {
        "model": "PINHOLE",
        "params": [
            float(intrinsic["fx"]),
            float(intrinsic["fy"]),
            float(intrinsic["cx"]),
            float(intrinsic["cy"]),
        ],
    }


def world_pose_to_poselib_camera_pose(world_pose: np.ndarray):
    require_poselib()
    rotation, translation = split_transform(world_pose)
    return poselib.CameraPose(rotation, translation)


def camera_extrinsics_for_generalized_pose(
    camera_ids: Iterable[str],
    camera_params: Dict[str, Dict],
    extrinsics: Dict[str, Dict],
) -> Tuple[List[object], List[dict]]:
    require_poselib()
    camera_ext = []
    camera_dicts = []
    for cam_id in camera_ids:
        t_world_cam = world_from_rgb_extrinsic(extrinsics[cam_id])
        camera_ext.append(world_pose_to_poselib_camera_pose(t_world_cam))
        camera_dicts.append(make_poselib_camera_dict(camera_params[cam_id]["RGB"]["intrinsic"]))
    return camera_ext, camera_dicts


def estimate_generalized_absolute_pose(
    correspondences: GeneralizedCorrespondenceSet,
    initial_pose: Optional[np.ndarray] = None,
    ransac_opt: Optional[dict] = None,
    bundle_opt: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    require_poselib()
    initial = None
    if initial_pose is not None:
        initial = world_pose_to_poselib_camera_pose(initial_pose)
    pose, info = poselib.estimate_generalized_absolute_pose(
        correspondences.points2d,
        correspondences.points3d,
        correspondences.camera_ext,
        correspondences.camera_dicts,
        {} if ransac_opt is None else ransac_opt,
        {} if bundle_opt is None else bundle_opt,
        initial,
    )
    return poselib_camera_pose_to_matrix(pose), info


def refine_generalized_absolute_pose(
    correspondences: GeneralizedCorrespondenceSet,
    initial_pose: np.ndarray,
    bundle_opt: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    require_poselib()
    pose, info = poselib.refine_generalized_absolute_pose(
        correspondences.points2d,
        correspondences.points3d,
        world_pose_to_poselib_camera_pose(initial_pose),
        correspondences.camera_ext,
        correspondences.camera_dicts,
        {} if bundle_opt is None else bundle_opt,
    )
    return poselib_camera_pose_to_matrix(pose), info


def poselib_camera_pose_to_matrix(camera_pose) -> np.ndarray:
    rotation = np.asarray(camera_pose.R, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(camera_pose.t, dtype=np.float64).reshape(3)
    return transform_matrix(rotation, translation)

