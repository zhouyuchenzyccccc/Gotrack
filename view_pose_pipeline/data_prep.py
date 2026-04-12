"""BOP dataset preparation from raw multi-camera data."""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import trimesh



def compute_models_info(mesh: trimesh.Trimesh) -> Dict:
    bounds = mesh.bounds
    mins, maxs = bounds[0], bounds[1]
    size = maxs - mins
    return {
        "1": {
            "diameter": float(np.linalg.norm(size)),
            "min_x": float(mins[0]), "min_y": float(mins[1]), "min_z": float(mins[2]),
            "size_x": float(size[0]), "size_y": float(size[1]), "size_z": float(size[2]),
        }
    }


def _copy_rgb(src: Path, dst: Path):
    if dst.exists():
        return
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read RGB: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)


def _copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_bytes(src.read_bytes())


def _rmtree(path: Path):
    for child in sorted(path.glob("**/*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    if path.exists():
        path.rmdir()


def ensure_multiview_bop_dataset(
    raw_data_dir: Path,
    bop_root: Path,
    base_dataset_name: str,
    cam_id: str,
    camera_params: Dict,
    mesh_path: Path,
    overwrite: bool,
) -> Tuple[str, int]:
    dataset_name = f"{base_dataset_name}_{cam_id}"
    dataset_dir = bop_root / dataset_name
    scene_dir = dataset_dir / "test" / "000001"
    targets_path = dataset_dir / "test_targets_bop19.json"
    rgb_src_dir = raw_data_dir / cam_id / "RGB"
    depth_src_dir = raw_data_dir / cam_id / "Depth"
    rgb_files = sorted(rgb_src_dir.glob("*.jpg"))
    frame_count = len(rgb_files)
    if frame_count == 0:
        raise FileNotFoundError(f"No RGB frames in {rgb_src_dir}")

    if overwrite and dataset_dir.exists():
        _rmtree(dataset_dir)

    if targets_path.exists():
        try:
            if len(json.loads(targets_path.read_text())) == frame_count:
                return dataset_name, frame_count
        except Exception:
            pass

    models_dir = dataset_dir / "models"
    rgb_out = scene_dir / "rgb"
    depth_out = scene_dir / "depth"
    for d in (models_dir, rgb_out, depth_out):
        d.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.export(models_dir / "obj_000001.ply")
    (models_dir / "models_info.json").write_text(json.dumps(compute_models_info(mesh), indent=2))

    first_rgb = cv2.imread(str(rgb_files[0]), cv2.IMREAD_COLOR)
    h, w = first_rgb.shape[:2]
    intr = camera_params[cam_id]["RGB"]["intrinsic"]
    (dataset_dir / "camera.json").write_text(json.dumps({
        "cx": float(intr["cx"]), "cy": float(intr["cy"]),
        "fx": float(intr["fx"]), "fy": float(intr["fy"]),
        "width": int(w), "height": int(h), "depth_scale": 1.0,
    }, indent=2))

    k = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], dtype=np.float64)
    scene_camera, scene_gt, scene_gt_info, targets = {}, {}, {}, []
    for rgb_path in rgb_files:
        fid = int(rgb_path.stem)
        depth_path = depth_src_dir / f"{fid:05d}.png"
        _copy_rgb(rgb_path, rgb_out / f"{fid:06d}.png")
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth: {depth_path}")
        _copy_file(depth_path, depth_out / f"{fid:06d}.png")
        scene_camera[str(fid)] = {"cam_K": k.reshape(-1).tolist(), "depth_scale": 1.0}
        scene_gt[str(fid)] = []
        scene_gt_info[str(fid)] = []
        targets.append({"scene_id": 1, "im_id": fid, "obj_id": 1, "inst_count": 1})

    (scene_dir / "scene_camera.json").write_text(json.dumps(scene_camera, indent=2))
    (scene_dir / "scene_gt.json").write_text(json.dumps(scene_gt, indent=2))
    (scene_dir / "scene_gt_info.json").write_text(json.dumps(scene_gt_info, indent=2))
    targets_path.write_text(json.dumps(targets, indent=2))
    return dataset_name, frame_count


def ensure_inference_results(
    repo_root: Path,
    dataset_name: str,
    frame_count: int,
    skip_inference: bool,
) -> Path:
    result_dir = repo_root / "results" / "inference_pose_pipeline" / "localization" / dataset_name
    if len(sorted(result_dir.glob("per_frame_refined_poses_*.json"))) >= frame_count:
        return result_dir
    if skip_inference:
        return result_dir
    cmd = [
        sys.executable, "-m", "scripts.inference_pose_estimation",
        f"dataset_name={dataset_name}", "mode=localization",
        f"user.root_dir={repo_root}", f"machine.root_dir={repo_root}",
        f"model.gotrack.checkpoint_path={repo_root / 'checkpoints' / 'gotrack_checkpoint.pt'}",
        "machine.trainer.strategy=ddp", "machine.trainer.devices=1",
    ]
    print(f"[data-prep] running inference for {dataset_name} ...", flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)
    return result_dir
