from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from .transforms import (
    invert_transform,
    rgb_from_depth_extrinsic,
    split_transform,
    world_from_depth_extrinsic,
    world_from_rgb_extrinsic,
)


class DepthCache:
    def __init__(
        self,
        raw_data_dir: Path,
        frame_id: int,
        camera_ids: Iterable[str],
        camera_params: Dict[str, Dict],
        extrinsics: Dict[str, Dict],
        max_depth_m: float,
    ):
        self.raw_data_dir = raw_data_dir
        self.frame_id = frame_id
        self.camera_ids = list(camera_ids)
        self.camera_params = camera_params
        self.extrinsics = extrinsics
        self.max_depth_m = max_depth_m
        self._depth_m: Dict[str, Optional[np.ndarray]] = {}
        self._points_world: Dict[Tuple[str, int], np.ndarray] = {}

    def _load_depth(self, cam_id: str) -> Optional[np.ndarray]:
        depth_path = self.raw_data_dir / cam_id / "Depth" / f"{self.frame_id:05d}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        return depth.astype(np.float32) / 1000.0

    def preload(self):
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(self.camera_ids)))) as pool:
            depth_maps = list(pool.map(self._load_depth, self.camera_ids))
        for cam_id, depth_m in zip(self.camera_ids, depth_maps):
            self._depth_m[cam_id] = depth_m

    def get_depth_m(self, cam_id: str) -> Optional[np.ndarray]:
        if cam_id not in self._depth_m:
            self._depth_m[cam_id] = self._load_depth(cam_id)
        return self._depth_m[cam_id]

    def get_points_world(self, cam_id: str, point_stride: int) -> np.ndarray:
        cache_key = (cam_id, point_stride)
        if cache_key not in self._points_world:
            self._points_world[cache_key] = depth_to_world_points(
                depth_m=self.get_depth_m(cam_id),
                cam_id=cam_id,
                camera_params=self.camera_params,
                extrinsics=self.extrinsics,
                point_stride=point_stride,
                max_depth_m=self.max_depth_m,
            )
        return self._points_world[cache_key]


def depth_to_world_points(
    depth_m: Optional[np.ndarray],
    cam_id: str,
    camera_params: Dict[str, Dict],
    extrinsics: Dict[str, Dict],
    point_stride: int,
    max_depth_m: float,
) -> np.ndarray:
    if depth_m is None:
        return np.empty((0, 3), dtype=np.float32)
    valid = (depth_m > 0.05) & (depth_m < max_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    ys, xs = np.nonzero(valid)
    if point_stride > 1:
        keep = np.arange(len(xs)) % point_stride == 0
        ys, xs = ys[keep], xs[keep]
    z = depth_m[ys, xs]
    intr = camera_params[cam_id]["rgb_to_depth"]["depth_intrinsic"]
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points_depth = np.stack([x, y, z], axis=1)
    t_world_depth = world_from_rgb_extrinsic(extrinsics[cam_id]) @ rgb_from_depth_extrinsic(
        extrinsics[cam_id]
    )
    r_world_depth, t_world_depth_vec = split_transform(t_world_depth)
    points_world = (r_world_depth @ points_depth.T).T + t_world_depth_vec[None, :]
    return points_world.astype(np.float32)


def project_points_to_depth_image(
    points_world: np.ndarray,
    camera_params: Dict[str, Dict],
    extrinsics: Dict[str, Dict],
    cam_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    t_world_depth = world_from_depth_extrinsic(extrinsics[cam_id])
    depth_from_world = invert_transform(t_world_depth)
    r_depth_world, t_depth_world = split_transform(depth_from_world)
    points_depth = (r_depth_world @ points_world.T).T + t_depth_world[None, :]
    intr = camera_params[cam_id]["rgb_to_depth"]["depth_intrinsic"]
    z = points_depth[:, 2]
    valid = z > 1e-4
    uv = np.full((len(points_depth), 2), -1.0, dtype=np.float64)
    uv[valid, 0] = points_depth[valid, 0] * intr["fx"] / z[valid] + intr["cx"]
    uv[valid, 1] = points_depth[valid, 1] * intr["fy"] / z[valid] + intr["cy"]
    return uv, points_depth

