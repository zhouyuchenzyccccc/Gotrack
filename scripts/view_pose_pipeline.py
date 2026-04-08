#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

try:
    import open3d as o3d
except Exception:
    o3d = None

matplotlib.use("agg")


DEFAULT_MESH_PATH = (
    "/home/ubuntu/WorkSpace/ZYC/FoundationPose/demo_data/electric_drill/mesh/Scan.ply"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="View GoTrack results. Supports single-view playback and multi-view fusion."
    )
    parser.add_argument("--bop-root", type=Path, required=True, help="Path to bop_datasets.")
    parser.add_argument("--dataset-name", type=str, required=True, help="Base dataset name.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Single-view results directory.")
    parser.add_argument("--scene-id", type=int, default=1, help="BOP scene id.")
    parser.add_argument("--fps", type=float, default=8.0, help="Playback FPS.")
    parser.add_argument("--watch", action="store_true", help="Wait for new frames.")
    parser.add_argument("--window-name", type=str, default="GoTrack Pose Viewer")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Raw multi-camera directory, e.g. /home/ubuntu/orbbec/src/sync/test/test/drill",
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=Path(DEFAULT_MESH_PATH),
        help="Object mesh used for BOP preparation and 3D visualization.",
    )
    parser.add_argument(
        "--camera-ids",
        type=str,
        default="00,01,02,03,04,05",
        help="Comma-separated camera ids for multi-view fusion.",
    )
    parser.add_argument(
        "--reference-camera",
        type=str,
        default="00",
        help="Camera id used for the 2D RGB/detection/coarse/refined panels.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only prepare data and visualize existing results.",
    )
    parser.add_argument(
        "--overwrite-prepared",
        action="store_true",
        help="Rebuild prepared BOP datasets even if they already exist.",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=2.0,
        help="Maximum depth in meters for fused point cloud rendering.",
    )
    parser.add_argument(
        "--point-stride",
        type=int,
        default=6,
        help="Stride used when subsampling depth pixels for fused point clouds.",
    )
    parser.add_argument(
        "--pose-inlier-thresh-m",
        type=float,
        default=0.12,
        help="Translation inlier threshold in meters for pose fusion.",
    )
    parser.add_argument(
        "--min-pose-score",
        type=float,
        default=0.6,
        help="Discard per-camera poses with score lower than this threshold.",
    )
    parser.add_argument(
        "--reject-large-jump",
        action="store_true",
        default=True,
        help="Reject very large frame-to-frame pose jumps and keep the previous pose.",
    )
    parser.add_argument(
        "--reject-translation-jump-m",
        type=float,
        default=0.20,
        help="If fused translation jumps more than this, keep the previous frame pose.",
    )
    parser.add_argument(
        "--reject-rotation-jump-deg",
        type=float,
        default=45.0,
        help="If fused rotation jumps more than this, keep the previous frame pose.",
    )
    parser.add_argument(
        "--interactive-3d",
        action="store_true",
        help="Open an interactive Open3D window for the fused point cloud.",
    )
    parser.add_argument(
        "--no-export-viewer-overlay",
        action="store_false",
        dest="export_viewer_overlay",
        help="Disable exporting fused mesh samples for the C++ sync viewer.",
    )
    return parser.parse_args()


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


def read_image(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def make_placeholder(shape, title: str, message: str) -> np.ndarray:
    image = np.full(shape, 245, dtype=np.uint8)
    cv2.putText(
        image,
        title,
        (18, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        message,
        (18, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (90, 90, 90),
        2,
        cv2.LINE_AA,
    )
    return image


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    title_h = 44
    canvas = np.full((image.shape[0] + title_h, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[title_h:] = image
    cv2.putText(
        canvas,
        title,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
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


def collect_available_frames(results_dir: Path) -> List[int]:
    frame_ids = set()
    for pattern in [
        "processed_detections_*.png",
        "vis_*_foundPose.png",
        "vis_*_goTrack.png",
    ]:
        for path in results_dir.glob(pattern):
            for part in path.stem.split("_"):
                if part.isdigit():
                    frame_ids.add(int(part))
                    break
    return sorted(frame_ids)


def build_frame_paths(results_dir: Path, frame_id: int) -> Dict[str, Path]:
    return {
        "detection": results_dir / f"processed_detections_{frame_id:06d}.png",
        "coarse": results_dir / f"vis_{frame_id:06d}_foundPose.png",
        "refined": results_dir / f"vis_{frame_id:06d}_goTrack.png",
    }


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


def default_results_dir(args) -> Path:
    if args.results_dir is not None:
        return args.results_dir.resolve()
    return (
        args.bop_root.parent
        / "results"
        / "inference_pose_pipeline"
        / "localization"
        / args.dataset_name
    ).resolve()


def run_single_view_viewer(args):
    results_dir = default_results_dir(args)
    rgb_dir = (
        args.bop_root.resolve()
        / args.dataset_name
        / "test"
        / f"{args.scene_id:06d}"
        / "rgb"
    )
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    delay_ms = max(1, int(1000 / max(args.fps, 0.1)))
    paused = False
    frame_ids = collect_available_frames(results_dir)
    index = 0
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    while True:
        if not frame_ids:
            if not args.watch:
                raise FileNotFoundError(f"No result frames found in {results_dir}")
            wait = np.full((360, 720, 3), 250, dtype=np.uint8)
            cv2.putText(
                wait,
                "Waiting for results...",
                (40, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (50, 50, 50),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(args.window_name, wait)
            key = cv2.waitKey(500) & 0xFF
            if key in (ord("q"), 27):
                break
            frame_ids = collect_available_frames(results_dir)
            continue

        frame_id = frame_ids[index]
        rgb_path = rgb_dir / f"{frame_id:06d}.png"
        rgb = read_image(rgb_path)
        if rgb is None:
            rgb = make_placeholder((480, 848, 3), "RGB", f"Missing: {rgb_path.name}")
        grid = build_single_view_grid(rgb, build_frame_paths(results_dir, frame_id), frame_id)
        draw_status(grid, paused)
        cv2.imshow(args.window_name, grid)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if handle_key(key):
            break
        paused, index, frame_ids = update_playback_state(
            key=key,
            paused=paused,
            index=index,
            frame_ids=frame_ids,
            results_dir=results_dir,
            watch=args.watch,
        )

    cv2.destroyAllWindows()


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
) -> Tuple[bool, int, List[int]]:
    if key == ord(" "):
        return (not paused, index, frame_ids)
    if key == ord("a"):
        return (True, max(0, index - 1), frame_ids)
    if key == ord("d"):
        return (True, min(len(frame_ids) - 1, index + 1), frame_ids)

    if not paused:
        latest_frame_ids = collect_available_frames(results_dir)
        if latest_frame_ids:
            frame_ids = latest_frame_ids
        if index < len(frame_ids) - 1:
            index += 1
        elif watch:
            time.sleep(0.1)
            frame_ids = collect_available_frames(results_dir)
    return (paused, index, frame_ids)


def parse_camera_ids(arg: str, raw_data_dir: Path) -> List[str]:
    if arg.strip():
        requested = [item.strip() for item in arg.split(",") if item.strip()]
    else:
        requested = []
    available = {path.name for path in raw_data_dir.iterdir() if path.is_dir()}
    camera_ids = [cam for cam in requested if cam in available]
    if not camera_ids:
        camera_ids = sorted([cam for cam in available if cam.isdigit()])[:6]
    return camera_ids


def camera_dataset_name(base_dataset_name: str, cam_id: str) -> str:
    return f"{base_dataset_name}_{cam_id}"


def load_json(path: Path):
    return json.loads(path.read_text())


def compute_models_info(mesh: trimesh.Trimesh) -> Dict[str, Dict[str, float]]:
    bounds = mesh.bounds
    mins = bounds[0]
    maxs = bounds[1]
    size = maxs - mins
    diameter = float(np.linalg.norm(size))
    return {
        "1": {
            "diameter": diameter,
            "min_x": float(mins[0]),
            "min_y": float(mins[1]),
            "min_z": float(mins[2]),
            "size_x": float(size[0]),
            "size_y": float(size[1]),
            "size_z": float(size[2]),
        }
    }


def rgb_intrinsic_matrix(camera_params: Dict[str, Dict], cam_id: str) -> np.ndarray:
    intr = camera_params[cam_id]["RGB"]["intrinsic"]
    return np.array(
        [
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def copy_rgb_to_png(src: Path, dst: Path):
    if dst.exists():
        return
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), image)


def ensure_multiview_bop_dataset(
    raw_data_dir: Path,
    bop_root: Path,
    base_dataset_name: str,
    cam_id: str,
    camera_params: Dict[str, Dict],
    mesh_path: Path,
    overwrite: bool,
) -> Tuple[str, int]:
    dataset_name = camera_dataset_name(base_dataset_name, cam_id)
    dataset_dir = bop_root / dataset_name
    scene_dir = dataset_dir / "test" / "000001"
    targets_path = dataset_dir / "test_targets_bop19.json"
    rgb_src_dir = raw_data_dir / cam_id / "RGB"
    depth_src_dir = raw_data_dir / cam_id / "Depth"
    rgb_files = sorted(rgb_src_dir.glob("*.jpg"))
    frame_count = len(rgb_files)
    if frame_count == 0:
        raise FileNotFoundError(f"No RGB frames found in {rgb_src_dir}")

    if overwrite and dataset_dir.exists():
        shutil_rmtree(dataset_dir)

    if targets_path.exists():
        try:
            existing_targets = json.loads(targets_path.read_text())
            if len(existing_targets) == frame_count:
                return dataset_name, frame_count
        except Exception:
            pass

    dataset_dir.mkdir(parents=True, exist_ok=True)
    models_dir = dataset_dir / "models"
    rgb_out = scene_dir / "rgb"
    depth_out = scene_dir / "depth"
    models_dir.mkdir(parents=True, exist_ok=True)
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh_dst = models_dir / "obj_000001.ply"
    mesh.export(mesh_dst)
    (models_dir / "models_info.json").write_text(
        json.dumps(compute_models_info(mesh), indent=2)
    )

    first_rgb = cv2.imread(str(rgb_files[0]), cv2.IMREAD_COLOR)
    h, w = first_rgb.shape[:2]
    intr = camera_params[cam_id]["RGB"]["intrinsic"]
    camera_json = {
        "cx": float(intr["cx"]),
        "cy": float(intr["cy"]),
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "width": int(w),
        "height": int(h),
        "depth_scale": 1.0,
    }
    (dataset_dir / "camera.json").write_text(json.dumps(camera_json, indent=2))

    scene_camera = {}
    scene_gt = {}
    scene_gt_info = {}
    targets = []
    k = rgb_intrinsic_matrix(camera_params, cam_id)
    for rgb_path in rgb_files:
        frame_id = int(rgb_path.stem)
        depth_path = depth_src_dir / f"{frame_id:05d}.png"
        copy_rgb_to_png(rgb_path, rgb_out / f"{frame_id:06d}.png")
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth frame: {depth_path}")
        copy_file(depth_path, depth_out / f"{frame_id:06d}.png")
        scene_camera[str(frame_id)] = {"cam_K": k.reshape(-1).tolist(), "depth_scale": 1.0}
        scene_gt[str(frame_id)] = []
        scene_gt_info[str(frame_id)] = []
        targets.append({"scene_id": 1, "im_id": frame_id, "obj_id": 1, "inst_count": 1})

    (scene_dir / "scene_camera.json").write_text(json.dumps(scene_camera, indent=2))
    (scene_dir / "scene_gt.json").write_text(json.dumps(scene_gt, indent=2))
    (scene_dir / "scene_gt_info.json").write_text(json.dumps(scene_gt_info, indent=2))
    targets_path.write_text(json.dumps(targets, indent=2))
    return dataset_name, frame_count


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    dst.write_bytes(src.read_bytes())


def shutil_rmtree(path: Path):
    for child in sorted(path.glob("**/*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    if path.exists():
        path.rmdir()


def expected_result_dir(repo_root: Path, dataset_name: str) -> Path:
    return repo_root / "results" / "inference_pose_pipeline" / "localization" / dataset_name


def ensure_inference_results(
    repo_root: Path,
    dataset_name: str,
    frame_count: int,
    skip_inference: bool,
):
    result_dir = expected_result_dir(repo_root, dataset_name)
    per_frame_files = sorted(result_dir.glob("per_frame_refined_poses_*.json"))
    if len(per_frame_files) >= frame_count:
        return result_dir
    if skip_inference:
        return result_dir

    cmd = [
        sys.executable,
        "-m",
        "scripts.inference_pose_estimation",
        f"dataset_name={dataset_name}",
        "mode=localization",
        f"user.root_dir={repo_root}",
        f"machine.root_dir={repo_root}",
        f"model.gotrack.checkpoint_path={repo_root / 'checkpoints' / 'gotrack_checkpoint.pt'}",
        "machine.trainer.strategy=ddp",
        "machine.trainer.devices=1",
    ]
    print(f"[multi-view] running inference for {dataset_name} ...", flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)
    return result_dir


def parse_pose_file(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None
    row = rows[0]
    return {
        "R": np.fromstring(row["R"], sep=" ", dtype=np.float64).reshape(3, 3),
        "t_m": np.fromstring(row["t"], sep=" ", dtype=np.float64) / 1000.0,
        "score": float(row["score"]),
    }


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
    color_from_depth = invert_transform(
        transform_matrix(
            np.asarray(c2d["rotation"], dtype=np.float64),
            np.asarray(c2d["translation"], dtype=np.float64) / 1000.0,
        )
    )
    return color_from_depth


def camera_pose_to_world_pose(pose: Dict[str, np.ndarray], extrinsics_entry: Dict) -> np.ndarray:
    t_rgb_m = pose["t_m"]
    T_rgb_m = transform_matrix(pose["R"], t_rgb_m)
    T_world_rgb = world_from_rgb_extrinsic(extrinsics_entry)
    return T_world_rgb @ T_rgb_m


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


def compute_pose_jump(
    previous_pose_world: np.ndarray, current_pose_world: np.ndarray
) -> Tuple[float, float]:
    prev_r, prev_t = split_transform(previous_pose_world)
    curr_r, curr_t = split_transform(current_pose_world)
    translation_jump = float(np.linalg.norm(curr_t - prev_t))
    rotation_jump = rotation_angle_deg(prev_r, curr_r)
    return translation_jump, rotation_jump


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

    translation_jump, rotation_jump = compute_pose_jump(
        previous_pose_world, current_pose_world
    )
    if (
        translation_jump <= translation_thresh_m
        and rotation_jump <= rotation_thresh_deg
    ):
        return current_pose_world, None

    print(
        "[pose-filter] frame "
        f"{frame_id:06d} rejected, "
        f"translation_jump={translation_jump:.4f} m, "
        f"rotation_jump={rotation_jump:.2f} deg. "
        "Using previous accepted pose."
    )
    return (
        previous_pose_world.copy(),
        f"replaced by prev | dt={translation_jump:.3f}m | dr={rotation_jump:.1f}deg",
    )


def fuse_world_poses(
    poses_world: Dict[str, Tuple[np.ndarray, float]],
    inlier_thresh_m: float,
) -> Tuple[Optional[np.ndarray], List[str]]:
    if not poses_world:
        return None, []
    cam_ids = list(poses_world.keys())
    transforms = [poses_world[cam_id][0] for cam_id in cam_ids]
    weights = np.asarray([max(poses_world[cam_id][1], 1e-6) for cam_id in cam_ids], dtype=np.float64)
    translations = np.asarray([tf[:3, 3] for tf in transforms], dtype=np.float64)
    weighted_center = np.average(translations, axis=0, weights=weights)
    distances = np.linalg.norm(translations - weighted_center[None, :], axis=1)
    inlier_mask = distances <= max(inlier_thresh_m, np.median(distances) * 1.5 if len(distances) > 1 else 0.0)
    if not np.any(inlier_mask):
        inlier_mask = np.ones(len(cam_ids), dtype=bool)
    inlier_cam_ids = [cam_ids[i] for i in range(len(cam_ids)) if inlier_mask[i]]
    inlier_transforms = [transforms[i] for i in range(len(transforms)) if inlier_mask[i]]
    inlier_weights = weights[inlier_mask]
    fused_t = np.average(
        np.asarray([tf[:3, 3] for tf in inlier_transforms]),
        axis=0,
        weights=inlier_weights,
    )
    quats = Rotation.from_matrix(np.asarray([tf[:3, :3] for tf in inlier_transforms])).as_quat()
    fused_q = weighted_average_quaternions(quats, inlier_weights)
    fused_R = Rotation.from_quat(fused_q).as_matrix()
    return transform_matrix(fused_R, fused_t), inlier_cam_ids


def load_depth_points_world(
    raw_data_dir: Path,
    cam_id: str,
    frame_id: int,
    camera_params: Dict[str, Dict],
    extrinsics: Dict[str, Dict],
    point_stride: int,
    max_depth_m: float,
) -> np.ndarray:
    depth_path = raw_data_dir / cam_id / "Depth" / f"{frame_id:05d}.png"
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
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
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points_depth = np.stack([x, y, z], axis=1)

    T_world_depth = world_from_rgb_extrinsic(extrinsics[cam_id]) @ rgb_from_depth_extrinsic(extrinsics[cam_id])
    R_world_depth, t_world_depth = split_transform(T_world_depth)
    points_world = (R_world_depth @ points_depth.T).T + t_world_depth[None, :]
    return points_world.astype(np.float32)


def world_from_depth_extrinsic(extrinsics_entry: Dict) -> np.ndarray:
    return world_from_rgb_extrinsic(extrinsics_entry) @ rgb_from_depth_extrinsic(
        extrinsics_entry
    )


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


def evaluate_candidate_pose_multiview(
    pose_world: np.ndarray,
    frame_id: int,
    camera_ids: List[str],
    raw_data_dir: Path,
    camera_params: Dict[str, Dict],
    extrinsics: Dict[str, Dict],
    mesh_points_m: np.ndarray,
    max_depth_m: float,
) -> Dict[str, object]:
    r_world_obj, t_world_obj = split_transform(pose_world)
    object_points_world = (r_world_obj @ mesh_points_m.T).T + t_world_obj[None, :]

    total_valid = 0
    total_inliers = 0
    residual_sum = 0.0
    per_view = {}
    for cam_id in camera_ids:
        depth_path = raw_data_dir / cam_id / "Depth" / f"{frame_id:05d}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            per_view[cam_id] = {"valid": 0, "inliers": 0, "mae_mm": float("inf")}
            continue
        depth_m = depth.astype(np.float32) / 1000.0
        h, w = depth.shape[:2]
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
        inliers = residual_m < 0.03
        valid_count = int(len(residual_m))
        inlier_count = int(np.count_nonzero(inliers))
        mae_mm = float(np.mean(residual_m) * 1000.0)
        total_valid += valid_count
        total_inliers += inlier_count
        residual_sum += float(np.sum(residual_m))
        per_view[cam_id] = {
            "valid": valid_count,
            "inliers": inlier_count,
            "mae_mm": mae_mm,
        }

    coverage = float(total_valid) / float(max(1, len(camera_ids) * len(mesh_points_m)))
    inlier_ratio = float(total_inliers) / float(max(1, total_valid))
    mean_residual_m = residual_sum / float(max(1, total_valid))
    # Lower is better. Coverage and inlier ratio reduce the cost.
    total_cost = mean_residual_m + 0.04 * (1.0 - inlier_ratio) + 0.02 * (1.0 - coverage)
    return {
        "cost": float(total_cost),
        "coverage": coverage,
        "inlier_ratio": inlier_ratio,
        "per_view": per_view,
    }


def select_and_refine_multiview_pose(
    candidate_poses_world: Dict[str, Tuple[np.ndarray, float]],
    frame_id: int,
    camera_ids: List[str],
    raw_data_dir: Path,
    camera_params: Dict[str, Dict],
    extrinsics: Dict[str, Dict],
    mesh_points_m: np.ndarray,
    max_depth_m: float,
) -> Tuple[Optional[np.ndarray], List[str], Dict[str, Dict[str, object]]]:
    if not candidate_poses_world:
        return None, [], {}

    evaluations: Dict[str, Dict[str, object]] = {}
    for cam_id, (pose_world, score) in candidate_poses_world.items():
        eval_info = evaluate_candidate_pose_multiview(
            pose_world=pose_world,
            frame_id=frame_id,
            camera_ids=camera_ids,
            raw_data_dir=raw_data_dir,
            camera_params=camera_params,
            extrinsics=extrinsics,
            mesh_points_m=mesh_points_m,
            max_depth_m=max_depth_m,
        )
        eval_info["score"] = float(score)
        evaluations[cam_id] = eval_info

    ranked = sorted(
        evaluations.items(),
        key=lambda item: (item[1]["cost"], -item[1]["inlier_ratio"], -item[1]["score"]),
    )
    best_cam_id = ranked[0][0]
    best_pose_world = candidate_poses_world[best_cam_id][0]
    best_cost = float(ranked[0][1]["cost"])

    selected_cam_ids = []
    selected_transforms = []
    selected_weights = []
    for cam_id, eval_info in ranked:
        if float(eval_info["cost"]) <= best_cost + 0.015:
            selected_cam_ids.append(cam_id)
            selected_transforms.append(candidate_poses_world[cam_id][0])
            selected_weights.append(
                max(1e-6, candidate_poses_world[cam_id][1])
                * max(0.05, float(eval_info["inlier_ratio"]))
                / max(float(eval_info["cost"]), 1e-6)
            )

    if not selected_transforms:
        selected_cam_ids = [best_cam_id]
        selected_transforms = [best_pose_world]
        selected_weights = [1.0]

    selected_weights_np = np.asarray(selected_weights, dtype=np.float64)
    fused_t = np.average(
        np.asarray([tf[:3, 3] for tf in selected_transforms], dtype=np.float64),
        axis=0,
        weights=selected_weights_np,
    )
    fused_q = weighted_average_quaternions(
        Rotation.from_matrix(
            np.asarray([tf[:3, :3] for tf in selected_transforms], dtype=np.float64)
        ).as_quat(),
        selected_weights_np,
    )
    fused_r = Rotation.from_quat(fused_q).as_matrix()
    refined_pose = transform_matrix(fused_r, fused_t)
    return refined_pose, selected_cam_ids, evaluations


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
            points_world[:, 0],
            points_world[:, 1],
            points_world[:, 2],
            s=0.4,
            c=np.full((len(points_world), 3), 0.55),
            alpha=0.35,
            depthshade=False,
        )

    if fused_pose_world is not None:
        R_world_obj, t_world_obj = split_transform(fused_pose_world)
        mesh_vis = (R_world_obj @ mesh_points_m.T).T + t_world_obj[None, :]
        ax.scatter(
            mesh_vis[:, 0],
            mesh_vis[:, 1],
            mesh_vis[:, 2],
            s=1.2,
            c=np.tile(np.array([[0.85, 0.15, 0.15]]), (len(mesh_vis), 1)),
            alpha=0.85,
            depthshade=False,
        )
        axis_len = 0.08
        colors = [(1, 0, 0), (0, 0.65, 0), (0, 0.2, 1)]
        for axis_id in range(3):
            axis_end = t_world_obj + R_world_obj[:, axis_id] * axis_len
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
            handle.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 255 60 60\n"
            )


def collect_frame_ids_from_raw(raw_data_dir: Path, camera_ids: List[str]) -> List[int]:
    frame_sets = []
    for cam_id in camera_ids:
        rgb_dir = raw_data_dir / cam_id / "RGB"
        frame_sets.append({int(path.stem) for path in rgb_dir.glob("*.jpg")})
    common = set.intersection(*frame_sets) if frame_sets else set()
    return sorted(common)


def run_multiview_viewer(args):
    raw_data_dir = args.raw_data_dir.resolve()
    repo_root = Path(__file__).resolve().parent.parent
    bop_root = args.bop_root.resolve()
    mesh_path = args.mesh_path.resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    camera_params = load_json(raw_data_dir / "camera_params.json")
    extrinsics = load_json(raw_data_dir / "extrinsics.json")
    camera_ids = [cam for cam in parse_camera_ids(args.camera_ids, raw_data_dir) if cam in extrinsics]
    if not camera_ids:
        raise ValueError("No valid camera ids found for multi-view fusion.")
    reference_camera = args.reference_camera if args.reference_camera in camera_ids else camera_ids[0]

    frame_counts = {}
    result_dirs = {}
    for cam_id in camera_ids:
        dataset_name, frame_count = ensure_multiview_bop_dataset(
            raw_data_dir=raw_data_dir,
            bop_root=bop_root,
            base_dataset_name=args.dataset_name,
            cam_id=cam_id,
            camera_params=camera_params,
            mesh_path=mesh_path,
            overwrite=args.overwrite_prepared,
        )
        frame_counts[cam_id] = frame_count
        result_dirs[cam_id] = ensure_inference_results(
            repo_root=repo_root,
            dataset_name=dataset_name,
            frame_count=frame_count,
            skip_inference=args.skip_inference,
        )

    frame_ids = collect_frame_ids_from_raw(raw_data_dir, camera_ids)
    if not frame_ids:
        raise FileNotFoundError("No synchronized frames found across the selected cameras.")

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(1.0 / 1000.0)
    mesh_points_m, _ = trimesh.sample.sample_surface(mesh, 2500)
    mesh_eval_points_m, _ = trimesh.sample.sample_surface(mesh, 900)
    overlay_dir = raw_data_dir / "gotrack_overlay"
    open3d_viewer = None
    if args.interactive_3d:
        if o3d is None:
            raise RuntimeError(
                "Open3D is not installed. Re-run in the gotrack environment or install open3d."
            )
        open3d_viewer = Open3DSceneViewer(f"{args.window_name} 3D")

    delay_ms = max(1, int(1000 / max(args.fps, 0.1)))
    paused = False
    index = 0
    previous_accepted_pose_world: Optional[np.ndarray] = None
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    while True:
        frame_id = frame_ids[index]
        per_cam_world_poses = {}
        per_cam_scores = {}
        all_candidate_poses_world = {}
        fallback_cam_id = None
        fused_points = []
        for cam_id in camera_ids:
            pose_path = result_dirs[cam_id] / f"per_frame_refined_poses_{frame_id:06d}.json"
            pose = parse_pose_file(pose_path)
            if pose is not None:
                per_cam_scores[cam_id] = pose["score"]
                pose_world = camera_pose_to_world_pose(pose, extrinsics[cam_id])
                all_candidate_poses_world[cam_id] = (pose_world, pose["score"])
                if pose["score"] >= args.min_pose_score:
                    per_cam_world_poses[cam_id] = (
                        pose_world,
                        pose["score"],
                    )
            fused_points.append(
                load_depth_points_world(
                    raw_data_dir=raw_data_dir,
                    cam_id=cam_id,
                    frame_id=frame_id,
                    camera_params=camera_params,
                    extrinsics=extrinsics,
                    point_stride=args.point_stride,
                    max_depth_m=args.max_depth_m,
                )
            )

        if not per_cam_world_poses and all_candidate_poses_world:
            fallback_cam_id = max(
                all_candidate_poses_world.keys(),
                key=lambda cam_id: all_candidate_poses_world[cam_id][1],
            )
            per_cam_world_poses = {
                fallback_cam_id: all_candidate_poses_world[fallback_cam_id]
            }

        fused_pose_world = None
        inlier_cam_ids = list(per_cam_world_poses.keys())
        selected_cam_ids: List[str] = []
        candidate_costs: Dict[str, float] = {}
        filter_status: Optional[str] = None
        if per_cam_world_poses:
            fused_pose_world, selected_cam_ids, evaluations = select_and_refine_multiview_pose(
                candidate_poses_world=per_cam_world_poses,
                frame_id=frame_id,
                camera_ids=camera_ids,
                raw_data_dir=raw_data_dir,
                camera_params=camera_params,
                extrinsics=extrinsics,
                mesh_points_m=mesh_eval_points_m,
                max_depth_m=args.max_depth_m,
            )
            candidate_costs = {
                cam_id: float(info["cost"]) for cam_id, info in evaluations.items()
            }
        fused_pose_world, filter_status = filter_large_pose_jump(
            frame_id=frame_id,
            current_pose_world=fused_pose_world,
            previous_pose_world=previous_accepted_pose_world,
            enabled=args.reject_large_jump,
            translation_thresh_m=args.reject_translation_jump_m,
            rotation_thresh_deg=args.reject_rotation_jump_deg,
        )
        if fused_pose_world is not None:
            previous_accepted_pose_world = fused_pose_world.copy()
        points_world = np.concatenate(fused_points, axis=0) if fused_points else np.empty((0, 3), dtype=np.float32)
        object_points_world = np.empty((0, 3), dtype=np.float32)
        object_origin = None
        object_rotation = None
        if fused_pose_world is not None:
            object_rotation, object_origin = split_transform(fused_pose_world)
            object_points_world = (
                (object_rotation @ mesh_points_m.T).T + object_origin[None, :]
            ).astype(np.float32)
        if args.export_viewer_overlay:
            write_viewer_overlay(
                overlay_dir=overlay_dir,
                frame_id=frame_id,
                object_points_world=object_points_world,
            )
        scene_image = render_fused_scene(
            points_world=points_world,
            fused_pose_world=fused_pose_world,
            mesh_points_m=mesh_points_m,
            title=(
                f"Fused 3D pose | views: {len(inlier_cam_ids)}"
                if filter_status is None
                else f"Fused 3D pose | views: {len(inlier_cam_ids)} | filtered"
            ),
        )
        if open3d_viewer is not None:
            open3d_viewer.update(
                points_world=points_world,
                object_points_world=object_points_world,
                object_origin=object_origin,
                object_rotation=object_rotation,
            )

        reference_rgb = read_image(raw_data_dir / reference_camera / "RGB" / f"{frame_id:05d}.jpg")
        if reference_rgb is None:
            reference_rgb = make_placeholder((480, 848, 3), "RGB", "Missing reference RGB")
        reference_paths = build_frame_paths(result_dirs[reference_camera], frame_id)
        grid = build_multiview_grid(
            raw_rgb=reference_rgb,
            frame_paths=reference_paths,
            frame_id=frame_id,
            fused_scene=scene_image,
            inlier_cam_ids=inlier_cam_ids,
            per_cam_scores=per_cam_scores,
            min_pose_score=args.min_pose_score,
            selected_cam_ids=selected_cam_ids,
            candidate_costs=candidate_costs,
            fallback_cam_id=fallback_cam_id,
            filter_status=filter_status,
        )
        draw_status(grid, paused)
        cv2.imshow(args.window_name, grid)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if handle_key(key):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("a"):
            index = max(0, index - 1)
            paused = True
            continue
        if key == ord("d"):
            index = min(len(frame_ids) - 1, index + 1)
            paused = True
            continue
        if not paused and index < len(frame_ids) - 1:
            index += 1

    cv2.destroyAllWindows()
    if open3d_viewer is not None:
        open3d_viewer.close()


def main():
    args = parse_args()
    if args.raw_data_dir is not None:
        run_multiview_viewer(args)
    else:
        run_single_view_viewer(args)


if __name__ == "__main__":
    main()
