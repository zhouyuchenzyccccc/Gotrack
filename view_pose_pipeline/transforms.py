from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    return mat


def split_transform(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return matrix[:3, :3], matrix[:3, 3]


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    rotation, translation = split_transform(matrix)
    inv_rotation = rotation.T
    inv_translation = -(inv_rotation @ translation)
    return transform_matrix(inv_rotation, inv_translation)


def world_from_rgb_extrinsic(extrinsics_entry: Dict) -> np.ndarray:
    rgb_from_world = transform_matrix(
        np.asarray(extrinsics_entry["rotation"], dtype=np.float64),
        np.asarray(extrinsics_entry["translation"], dtype=np.float64),
    )
    return invert_transform(rgb_from_world)


def rgb_from_depth_extrinsic(extrinsics_entry: Dict) -> np.ndarray:
    rgb_to_depth = extrinsics_entry["rgb_to_depth"]
    if "d2c_extrinsic" in rgb_to_depth:
        d2c = rgb_to_depth["d2c_extrinsic"]
        return transform_matrix(
            np.asarray(d2c["rotation"], dtype=np.float64),
            np.asarray(d2c["translation"], dtype=np.float64) / 1000.0,
        )
    c2d = rgb_to_depth["c2d_extrinsic"]
    return invert_transform(
        transform_matrix(
            np.asarray(c2d["rotation"], dtype=np.float64),
            np.asarray(c2d["translation"], dtype=np.float64) / 1000.0,
        )
    )


def world_from_depth_extrinsic(extrinsics_entry: Dict) -> np.ndarray:
    return world_from_rgb_extrinsic(extrinsics_entry) @ rgb_from_depth_extrinsic(
        extrinsics_entry
    )


def camera_pose_to_world_pose(pose, extrinsics_entry: Dict) -> np.ndarray:
    t_rgb_m = pose["t_m"]
    t_rgb_pose = transform_matrix(pose["R"], t_rgb_m)
    t_world_rgb = world_from_rgb_extrinsic(extrinsics_entry)
    return t_world_rgb @ t_rgb_pose


def weighted_average_quaternions(quaternions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for quat, weight in zip(quaternions, weights):
        q = quat / np.linalg.norm(quat)
        if q[3] < 0:
            q = -q
        accumulator += weight * np.outer(q, q)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    quat = eigenvectors[:, np.argmax(eigenvalues)]
    quat /= np.linalg.norm(quat)
    return quat


def rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    cos_theta = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def average_transforms(transforms: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    fused_t = np.average(
        np.asarray([tf[:3, 3] for tf in transforms], dtype=np.float64),
        axis=0,
        weights=weights,
    )
    fused_q = weighted_average_quaternions(
        Rotation.from_matrix(
            np.asarray([tf[:3, :3] for tf in transforms], dtype=np.float64)
        ).as_quat(),
        weights,
    )
    fused_r = Rotation.from_quat(fused_q).as_matrix()
    return transform_matrix(fused_r, fused_t)

