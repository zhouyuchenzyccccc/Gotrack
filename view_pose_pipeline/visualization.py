"""Rendering and panel-building utilities."""
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .transforms import split_transform

matplotlib.use("agg")


# ---------------------------------------------------------------------------
# Basic image helpers
# ---------------------------------------------------------------------------

def read_image(path) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def make_placeholder(shape: Tuple, title: str, message: str) -> np.ndarray:
    img = np.full(shape, 245, dtype=np.uint8)
    cv2.putText(img, title,   (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40),  2, cv2.LINE_AA)
    cv2.putText(img, message, (18, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(90, 90, 90),  2, cv2.LINE_AA)
    return img


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    th = 44
    canvas = np.full((image.shape[0] + th, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[th:] = image
    cv2.putText(canvas, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def fit_to_size(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    tw, th = size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def stack_grid(images: Sequence[np.ndarray], cols: int) -> np.ndarray:
    rows = []
    for start in range(0, len(images), cols):
        row = list(images[start:start + cols])
        if len(row) < cols:
            row += [np.full_like(row[0], 255) for _ in range(cols - len(row))]
        rows.append(np.hstack(row))
    return np.vstack(rows)


def draw_status(image: np.ndarray, paused: bool):
    status = "paused" if paused else "playing"
    cv2.putText(
        image, f"{status} | q quit | space pause | a/d prev/next",
        (18, image.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# 3-D scene rendering (matplotlib → BGR image)
# ---------------------------------------------------------------------------

def _set_equal_axis(ax, points_world: np.ndarray, pose_world: Optional[np.ndarray]):
    if points_world.size:
        mins, maxs = points_world.min(axis=0), points_world.max(axis=0)
    elif pose_world is not None:
        c = pose_world[:3, 3]
        mins, maxs = c - 0.2, c + 0.2
    else:
        mins, maxs = np.array([-0.5, -0.5, 0.0]), np.array([0.5, 0.5, 1.0])

    if pose_world is not None:
        c = pose_world[:3, 3]
        mins = np.minimum(mins, c - 0.15)
        maxs = np.maximum(maxs, c + 0.15)

    center = (mins + maxs) / 2.0
    r = max(np.max(maxs - mins) / 2.0, 0.2)
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(max(0.0, center[2] - r), center[2] + r)


def render_fused_scene(
    points_world: np.ndarray,
    fused_pose_world: Optional[np.ndarray],
    mesh_points_m: np.ndarray,
    title: str,
) -> np.ndarray:
    fig = plt.figure(figsize=(7, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    if points_world.size:
        ax.scatter(
            points_world[:, 0], points_world[:, 1], points_world[:, 2],
            s=0.4, c=np.full((len(points_world), 3), 0.55), alpha=0.35, depthshade=False,
        )

    if fused_pose_world is not None:
        R, t = split_transform(fused_pose_world)
        mv = (R @ mesh_points_m.T).T + t[None, :]
        ax.scatter(
            mv[:, 0], mv[:, 1], mv[:, 2],
            s=1.2, c=np.tile([[0.85, 0.15, 0.15]], (len(mv), 1)), alpha=0.85, depthshade=False,
        )
        for i, color in enumerate([(1, 0, 0), (0, 0.65, 0), (0, 0.2, 1)]):
            end = t + R[:, i] * 0.08
            ax.plot([t[0], end[0]], [t[1], end[1]], [t[2], end[2]], color=color, linewidth=2.5)

    ax.set_title(title)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.view_init(elev=24, azim=-68)
    _set_equal_axis(ax, points_world, fused_pose_world)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Info panel and grid builders
# ---------------------------------------------------------------------------

def build_info_panel(
    shape: Tuple,
    frame_id: int,
    consensus_cam_ids: List[str],
    per_cam_scores: Dict[str, float],
    consistency_scores: Dict[str, float],
    min_pose_score: float,
    filter_status: Optional[str] = None,
) -> np.ndarray:
    panel = np.full(shape, 248, dtype=np.uint8)
    lines = [
        f"frame: {frame_id:06d}",
        f"consensus views: {', '.join(consensus_cam_ids) if consensus_cam_ids else 'none'}",
        f"min score: {min_pose_score:.2f}",
    ]
    if filter_status:
        lines.append(f"filter: {filter_status}")
    for cam_id in sorted(per_cam_scores):
        state = "use" if per_cam_scores[cam_id] >= min_pose_score else "skip"
        cons = consistency_scores.get(cam_id, float("nan"))
        lines.append(f"{cam_id}: det={per_cam_scores[cam_id]:.3f} [{state}] cons={cons:.3f}")
    y = 44
    for idx, line in enumerate(lines):
        scale = 0.8 if idx == 0 else 0.62
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), 2, cv2.LINE_AA)
        y += 32
    return panel


def build_multiview_grid(
    raw_rgb: np.ndarray,
    frame_paths: Dict,
    frame_id: int,
    fused_scene: np.ndarray,
    consensus_cam_ids: List[str],
    per_cam_scores: Dict[str, float],
    consistency_scores: Dict[str, float],
    min_pose_score: float,
    filter_status: Optional[str] = None,
) -> np.ndarray:
    base_shape = raw_rgb.shape
    panels = [
        ("RGB",            raw_rgb),
        ("Detection",      read_image(frame_paths["detection"])),
        ("Coarse Pose",    read_image(frame_paths["coarse"])),
        ("Refined Pose",   read_image(frame_paths["refined"])),
        ("Fused 3D Scene", fused_scene),
        ("Fusion Info",    build_info_panel(
            fused_scene.shape, frame_id, consensus_cam_ids,
            per_cam_scores, consistency_scores, min_pose_score, filter_status,
        )),
    ]
    titled = []
    for title, img in panels:
        if img is None:
            img = make_placeholder(base_shape, title, "Waiting for result...")
        titled.append(add_title(img, f"{title} | frame {frame_id:06d}"))
    mh = max(i.shape[0] for i in titled)
    mw = max(i.shape[1] for i in titled)
    return stack_grid([fit_to_size(i, (mw, mh)) for i in titled], cols=3)


def build_single_view_grid(rgb: np.ndarray, paths: Dict, frame_id: int) -> np.ndarray:
    panels = {
        "RGB": rgb,
        "Detection": read_image(paths["detection"]),
        "Coarse Pose": read_image(paths["coarse"]),
        "Refined Pose": read_image(paths["refined"]),
    }
    titled = []
    for title, img in panels.items():
        if img is None:
            img = make_placeholder(rgb.shape, title, "Waiting for result...")
        titled.append(add_title(img, f"{title} | frame {frame_id:06d}"))
    mh = max(i.shape[0] for i in titled)
    mw = max(i.shape[1] for i in titled)
    return stack_grid([fit_to_size(i, (mw, mh)) for i in titled], cols=2)
