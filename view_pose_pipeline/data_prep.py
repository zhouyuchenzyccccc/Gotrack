from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import trimesh

from .io_utils import copy_file, copy_rgb_to_png, shutil_rmtree


def parse_camera_ids(arg: str, raw_data_dir: Path) -> List[str]:
    requested = [item.strip() for item in arg.split(",") if item.strip()] if arg.strip() else []
    available = {path.name for path in raw_data_dir.iterdir() if path.is_dir()}
    camera_ids = [cam for cam in requested if cam in available]
    if not camera_ids:
        camera_ids = sorted([cam for cam in available if cam.isdigit()])[:6]
    return camera_ids


def camera_dataset_name(base_dataset_name: str, cam_id: str) -> str:
    return f"{base_dataset_name}_{cam_id}"


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


def expected_result_dir(repo_root: Path, dataset_name: str) -> Path:
    return repo_root / "results" / "inference_pose_pipeline" / "localization" / dataset_name


def ensure_inference_results(
    repo_root: Path,
    dataset_name: str,
    frame_count: int,
    skip_inference: bool,
) -> Path:
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

