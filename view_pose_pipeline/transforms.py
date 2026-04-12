"""SE3 rigid body transform utilities."""
from typing import Dict, Tuple

import numpy as np

def transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    return mat


def split_transform(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return matrix[:3, :3], matrix[:3, 3]


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    R, t = split_transform(matrix)
    Ri = R.T
    return transform_matrix(Ri, -(Ri @ t))


def world_from_rgb_extrinsic(entry: Dict) -> np.ndarray:
    rgb_from_world = transform_matrix(
        np.asarray(entry["rotation"], dtype=np.float64),
        np.asarray(entry["translation"], dtype=np.float64),
    )
    return invert_transform(rgb_from_world)


def rgb_from_depth_extrinsic(entry: Dict) -> np.ndarray:
    r2d = entry["rgb_to_depth"]
    if "d2c_extrinsic" in r2d:
        d2c = r2d["d2c_extrinsic"]
        return transform_matrix(
            np.asarray(d2c["rotation"], dtype=np.float64),
            np.asarray(d2c["translation"], dtype=np.float64) / 1000.0,
        )
    c2d = r2d["c2d_extrinsic"]
    return invert_transform(
        transform_matrix(
            np.asarray(c2d["rotation"], dtype=np.float64),
            np.asarray(c2d["translation"], dtype=np.float64) / 1000.0,
        )
    )


def world_from_depth_extrinsic(entry: Dict) -> np.ndarray:
    return world_from_rgb_extrinsic(entry) @ rgb_from_depth_extrinsic(entry)


def camera_pose_to_world_pose(pose: Dict, extrinsics_entry: Dict) -> np.ndarray:
    T_rgb_m = transform_matrix(pose["R"], pose["t_m"])
    T_world_rgb = world_from_rgb_extrinsic(extrinsics_entry)
    return T_world_rgb @ T_rgb_m


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    rel = R_a.T @ R_b
    cos_theta = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def weighted_average_quaternions(quaternions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    acc = np.zeros((4, 4), dtype=np.float64)
    for q, w in zip(quaternions, weights):
        q = q / np.linalg.norm(q)
        if q[3] < 0:
            q = -q
        acc += w * np.outer(q, q)
    _, vecs = np.linalg.eigh(acc)
    q = vecs[:, -1]
    return q / np.linalg.norm(q)
