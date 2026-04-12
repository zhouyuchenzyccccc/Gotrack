from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import read_image
from .transforms import split_transform

try:
    import open3d as o3d
except Exception:
    o3d = None

matplotlib.use("agg")


class Open3DSceneViewer:
    def __init__(self, window_name: str):
        if o3d is None:
            raise RuntimeError("Open3D is not available in the current environment.")
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(window_name=window_name, width=1280, height=900)
        render_option = self.visualizer.get_render_option()
        render_option.background_color = np.asarray([0.97, 0.97, 0.97])
        render_option.point_size = 2.0
        self.point_cloud = o3d.geometry.PointCloud()
        self.object_cloud = o3d.geometry.PointCloud()
        self.object_axes = o3d.geometry.LineSet()
        self.is_initialized = False

    def update(
        self,
        points_world: np.ndarray,
        object_points_world: np.ndarray,
        object_origin: Optional[np.ndarray],
        object_rotation: Optional[np.ndarray],
    ):
        points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        object_points_world = np.asarray(object_points_world, dtype=np.float64).reshape(-1, 3)
        self.point_cloud.points = o3d.utility.Vector3dVector(points_world.astype(np.float64))
        self.point_cloud.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.55, 0.55, 0.55]], dtype=np.float64), (len(points_world), 1))
        )
        self.object_cloud.points = o3d.utility.Vector3dVector(
            object_points_world.astype(np.float64)
        )
        self.object_cloud.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.88, 0.18, 0.18]], dtype=np.float64), (len(object_points_world), 1))
        )

        axes_points = np.zeros((4, 3), dtype=np.float64)
        axes_lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
        axes_colors = np.array([[1, 0, 0], [0, 0.7, 0], [0, 0.2, 1]], dtype=np.float64)
        if object_origin is not None and object_rotation is not None:
            axes_points[0] = object_origin
            axis_len = 0.08
            axes_points[1] = object_origin + object_rotation[:, 0] * axis_len
            axes_points[2] = object_origin + object_rotation[:, 1] * axis_len
            axes_points[3] = object_origin + object_rotation[:, 2] * axis_len
        self.object_axes.points = o3d.utility.Vector3dVector(axes_points)
        self.object_axes.lines = o3d.utility.Vector2iVector(axes_lines)
        self.object_axes.colors = o3d.utility.Vector3dVector(axes_colors)

        if not self.is_initialized:
            self.visualizer.add_geometry(self.point_cloud)
            self.visualizer.add_geometry(self.object_cloud)
            self.visualizer.add_geometry(self.object_axes)
            self.is_initialized = True
        else:
            self.visualizer.update_geometry(self.point_cloud)
            self.visualizer.update_geometry(self.object_cloud)
            self.visualizer.update_geometry(self.object_axes)

        self.visualizer.poll_events()
        self.visualizer.update_renderer()

    def close(self):
        self.visualizer.destroy_window()


def make_placeholder(shape, title: str, message: str) -> np.ndarray:
    image = np.full(shape, 245, dtype=np.uint8)
    cv2.putText(image, title, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(image, message, (18, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 2, cv2.LINE_AA)
    return image


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    title_h = 44
    canvas = np.full((image.shape[0] + title_h, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[title_h:] = image
    cv2.putText(canvas, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def fit_to_size(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def stack_grid(images: Sequence[np.ndarray], cols: int) -> np.ndarray:
    rows = []
    for start in range(0, len(images), cols):
        row = images[start : start + cols]
        if len(row) < cols:
            row = list(row) + [np.full_like(row[0], 255) for _ in range(cols - len(row))]
        rows.append(np.hstack(row))
    return np.vstack(rows)


def build_single_view_grid(rgb: np.ndarray, paths: Dict[str, Path], frame_id: int) -> np.ndarray:
    base_shape = rgb.shape
    panels = {
        "RGB": rgb,
        "Detection": read_image(paths["detection"]),
        "Coarse Pose": read_image(paths["coarse"]),
        "Refined Pose": read_image(paths["refined"]),
    }
    titled = []
    for title, image in panels.items():
        if image is None:
            image = make_placeholder(base_shape, title, "Waiting for result...")
        image = add_title(image, f"{title} | frame {frame_id:06d}")
        titled.append(image)
    max_h = max(img.shape[0] for img in titled)
    max_w = max(img.shape[1] for img in titled)
    fitted = [fit_to_size(img, (max_w, max_h)) for img in titled]
    return stack_grid(fitted, cols=2)


def draw_status(image: np.ndarray, paused: bool):
    status = "paused" if paused else "playing"
    cv2.putText(
        image,
        f"{status} | q quit | space pause | a/d prev/next",
        (18, image.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )


def handle_key(key: int) -> bool:
    return key in (ord("q"), 27)


def update_playback_state(
    key: int,
    paused: bool,
    index: int,
    frame_ids: List[int],
    results_dir: Path,
    watch: bool,
    collect_available_frames_fn,
) -> Tuple[bool, int, List[int]]:
    if key == ord(" "):
        return (not paused, index, frame_ids)
    if key == ord("a"):
        return (True, max(0, index - 1), frame_ids)
    if key == ord("d"):
        return (True, min(len(frame_ids) - 1, index + 1), frame_ids)

    if not paused:
        latest_frame_ids = collect_available_frames_fn(results_dir)
        if latest_frame_ids:
            frame_ids = latest_frame_ids
        if index < len(frame_ids) - 1:
            index += 1
        elif watch:
            time.sleep(0.1)
            frame_ids = collect_available_frames_fn(results_dir)
    return (paused, index, frame_ids)


def render_fused_scene(
    points_world: np.ndarray,
    fused_pose_world: Optional[np.ndarray],
    mesh_points_m: np.ndarray,
    title: str,
) -> np.ndarray:
    fig = plt.figure(figsize=(7, 6), dpi=110)
    ax = fig.add_subplot(111, projection="3d")

    if points_world.size:
        ax.scatter(
            points_world[:, 0],
            points_world[:, 1],
            points_world[:, 2],
            s=0.35,
            c=np.full((len(points_world), 3), 0.55),
            alpha=0.30,
            depthshade=False,
        )

    if fused_pose_world is not None:
        r_world_obj, t_world_obj = split_transform(fused_pose_world)
        mesh_vis = (r_world_obj @ mesh_points_m.T).T + t_world_obj[None, :]
        ax.scatter(
            mesh_vis[:, 0],
            mesh_vis[:, 1],
            mesh_vis[:, 2],
            s=1.0,
            c=np.tile(np.array([[0.85, 0.15, 0.15]]), (len(mesh_vis), 1)),
            alpha=0.85,
            depthshade=False,
        )
        axis_len = 0.08
        colors = [(1, 0, 0), (0, 0.65, 0), (0, 0.2, 1)]
        for axis_id in range(3):
            axis_end = t_world_obj + r_world_obj[:, axis_id] * axis_len
            ax.plot(
                [t_world_obj[0], axis_end[0]],
                [t_world_obj[1], axis_end[1]],
                [t_world_obj[2], axis_end[2]],
                color=colors[axis_id],
                linewidth=2.5,
            )

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=24, azim=-68)
    set_equal_axis(ax, points_world, fused_pose_world)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def set_equal_axis(ax, points_world: np.ndarray, fused_pose_world: Optional[np.ndarray]):
    if points_world.size:
        mins = points_world.min(axis=0)
        maxs = points_world.max(axis=0)
    elif fused_pose_world is not None:
        center = fused_pose_world[:3, 3]
        mins = center - 0.2
        maxs = center + 0.2
    else:
        mins = np.array([-0.5, -0.5, 0.0])
        maxs = np.array([0.5, 0.5, 1.0])

    if fused_pose_world is not None:
        center = fused_pose_world[:3, 3]
        mins = np.minimum(mins, center - 0.15)
        maxs = np.maximum(maxs, center + 0.15)

    center = (mins + maxs) / 2.0
    radius = max(np.max(maxs - mins) / 2.0, 0.2)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)


def build_info_panel(
    shape: Tuple[int, int, int],
    frame_id: int,
    inlier_cam_ids: List[str],
    per_cam_scores: Dict[str, float],
    min_pose_score: float,
    selected_cam_ids: Optional[List[str]] = None,
    candidate_costs: Optional[Dict[str, float]] = None,
    fallback_cam_id: Optional[str] = None,
    filter_status: Optional[str] = None,
) -> np.ndarray:
    panel = np.full(shape, 248, dtype=np.uint8)
    lines = [
        f"frame: {frame_id:06d}",
        f"fused views: {', '.join(inlier_cam_ids) if inlier_cam_ids else 'none'}",
        f"min score: {min_pose_score:.2f}",
    ]
    if filter_status:
        lines.append(f"filter: {filter_status}")
    if selected_cam_ids:
        lines.append(f"selected: {', '.join(selected_cam_ids)}")
    if fallback_cam_id:
        lines.append(f"fallback: {fallback_cam_id}")
    for cam_id in sorted(per_cam_scores):
        state = "use" if per_cam_scores[cam_id] >= min_pose_score else "skip"
        if candidate_costs and cam_id in candidate_costs:
            lines.append(
                f"{cam_id}: {per_cam_scores[cam_id]:.3f} [{state}] cost={candidate_costs[cam_id]:.4f}"
            )
        else:
            lines.append(f"{cam_id}: {per_cam_scores[cam_id]:.3f} [{state}]")
    y = 44
    for idx, line in enumerate(lines):
        scale = 0.8 if idx == 0 else 0.65
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), 2, cv2.LINE_AA)
        y += 34
    return panel


def build_multiview_grid(
    raw_rgb: np.ndarray,
    frame_paths: Dict[str, Path],
    frame_id: int,
    fused_scene: np.ndarray,
    inlier_cam_ids: List[str],
    per_cam_scores: Dict[str, float],
    min_pose_score: float,
    selected_cam_ids: Optional[List[str]] = None,
    candidate_costs: Optional[Dict[str, float]] = None,
    fallback_cam_id: Optional[str] = None,
    filter_status: Optional[str] = None,
) -> np.ndarray:
    base_shape = raw_rgb.shape
    panels = [
        ("RGB", raw_rgb),
        ("Detection", read_image(frame_paths["detection"])),
        ("Coarse Pose", read_image(frame_paths["coarse"])),
        ("Refined Pose", read_image(frame_paths["refined"])),
        ("Fused Point Cloud", fused_scene),
        (
            "Fusion Info",
            build_info_panel(
                fused_scene.shape,
                frame_id,
                inlier_cam_ids,
                per_cam_scores,
                min_pose_score,
                selected_cam_ids=selected_cam_ids,
                candidate_costs=candidate_costs,
                fallback_cam_id=fallback_cam_id,
                filter_status=filter_status,
            ),
        ),
    ]
    titled = []
    for title, image in panels:
        if image is None:
            image = make_placeholder(base_shape, title, "Waiting for result...")
        image = add_title(image, f"{title} | frame {frame_id:06d}")
        titled.append(image)
    max_h = max(img.shape[0] for img in titled)
    max_w = max(img.shape[1] for img in titled)
    fitted = [fit_to_size(img, (max_w, max_h)) for img in titled]
    return stack_grid(fitted, cols=3)


def write_viewer_overlay(
    overlay_dir: Path,
    frame_id: int,
    object_points_world: np.ndarray,
):
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_path = overlay_dir / f"frame_{frame_id:05d}.xyzrgb"
    if object_points_world.size == 0:
        if out_path.exists():
            out_path.unlink()
        return
    with out_path.open("w", encoding="utf-8") as handle:
        for p in object_points_world:
            handle.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 255 60 60\n")

