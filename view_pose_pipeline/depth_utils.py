"""Depth image loading, caching, and 3D projection utilities."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .transforms import invert_transform, split_transform, world_from_depth_extrinsic


class DepthCache:
    """Cache depth images to avoid redundant disk reads within a frame."""

    def __init__(self, max_entries: int = 60):
        self._cache: Dict[Tuple[str, int], Optional[np.ndarray]] = {}
        self._order: List[Tuple[str, int]] = []
        self._max = max_entries

    def get(self, cam_id: str, frame_id: int, depth_path: Path) -> Optional[np.ndarray]:
        key = (cam_id, frame_id)
        if key not in self._cache:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            self._cache[key] = depth
            self._order.append(key)
            if len(self._order) > self._max:
                old = self._order.pop(0)
                self._cache.pop(old, None)
        return self._cache[key]

    def preload_frame(self, frame_id: int, camera_ids: List[str], raw_data_dir: Path):
        """Parallel preload of all cameras' depth for one frame."""
        def _load(cam_id: str):
            path = raw_data_dir / cam_id / "Depth" / f"{frame_id:05d}.png"
            self.get(cam_id, frame_id, path)

        with ThreadPoolExecutor(max_workers=min(len(camera_ids), 8)) as pool:
            list(pool.map(_load, camera_ids))


def project_mesh_to_depth(
    object_points_world: np.ndarray,
    camera_params: Dict,
    extrinsics: Dict,
    cam_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world-space object points into a camera's depth image space.

    Returns (uv [N,2], points_in_depth_frame [N,3]).
    """
    T_world_depth = world_from_depth_extrinsic(extrinsics[cam_id])
    depth_from_world = invert_transform(T_world_depth)
    R, t = split_transform(depth_from_world)
    pts_d = (R @ object_points_world.T).T + t[None, :]

    intr = camera_params[cam_id]["rgb_to_depth"]["depth_intrinsic"]
    z = pts_d[:, 2]
    valid = z > 1e-4
    uv = np.full((len(pts_d), 2), -1.0, dtype=np.float64)
    uv[valid, 0] = pts_d[valid, 0] * intr["fx"] / z[valid] + intr["cx"]
    uv[valid, 1] = pts_d[valid, 1] * intr["fy"] / z[valid] + intr["cy"]
    return uv, pts_d


def load_depth_points_world(
    raw_data_dir: Path,
    cam_id: str,
    frame_id: int,
    camera_params: Dict,
    extrinsics: Dict,
    point_stride: int,
    max_depth_m: float,
    depth_cache: Optional[DepthCache] = None,
) -> np.ndarray:
    depth_path = raw_data_dir / cam_id / "Depth" / f"{frame_id:05d}.png"
    depth = (
        depth_cache.get(cam_id, frame_id, depth_path)
        if depth_cache is not None
        else cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    )
    if depth is None:
        return np.empty((0, 3), dtype=np.float32)

    depth_m = depth.astype(np.float32) / 1000.0
    valid = (depth_m > 0.05) & (depth_m < max_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    ys, xs = np.nonzero(valid)
    if point_stride > 1:
        keep = np.arange(len(xs)) % point_stride == 0
        ys, xs = ys[keep], xs[keep]
    z = depth_m[ys, xs]

    intr = camera_params[cam_id]["rgb_to_depth"]["depth_intrinsic"]
    x = (xs.astype(np.float32) - intr["cx"]) * z / intr["fx"]
    y = (ys.astype(np.float32) - intr["cy"]) * z / intr["fy"]
    pts_depth = np.stack([x, y, z], axis=1)

    T_world_depth = world_from_depth_extrinsic(extrinsics[cam_id])
    R, t = split_transform(T_world_depth)
    return ((R @ pts_depth.T).T + t[None, :]).astype(np.float32)
