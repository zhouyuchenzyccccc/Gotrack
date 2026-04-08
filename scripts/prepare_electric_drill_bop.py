#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a FoundationPose-style electric_drill demo into a minimal BOP dataset."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="electric_drill")
    parser.add_argument("--mesh-path", type=Path, default=None)
    return parser.parse_args()


def read_camera_matrix(path: Path):
    values = np.loadtxt(path, dtype=np.float64).reshape(3, 3)
    return values


def build_models_info(mesh_path: Path):
    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    diameter = float(np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1).max())
    return {
        "1": {
            "diameter": diameter,
            "min_x": float(mins[0]),
            "min_y": float(mins[1]),
            "min_z": float(mins[2]),
            "size_x": float(maxs[0] - mins[0]),
            "size_y": float(maxs[1] - mins[1]),
            "size_z": float(maxs[2] - mins[2]),
        }
    }


def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    source = args.source.resolve()
    dataset_dir = (args.output_root / args.dataset_name).resolve()
    scene_dir = dataset_dir / "test" / "000001"
    rgb_out = scene_dir / "rgb"
    depth_out = scene_dir / "depth"
    mask_out = scene_dir / "mask"
    mask_visib_out = scene_dir / "mask_visib"
    models_dir = dataset_dir / "models"

    rgb_files = sorted((source / "rgb").glob("*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found under {source / 'rgb'}")

    depth_files = {p.stem: p for p in (source / "depth").glob("*.png") if "visualize" not in p.name}
    mask_files = {p.stem: p for p in (source / "masks").glob("*.png")}

    mesh_path = args.mesh_path
    if mesh_path is None:
        candidates = [
            source / "mesh" / "Scan.ply",
            source / "mesh" / "Scan.obj",
        ]
        mesh_path = next((p for p in candidates if p.exists()), None)
    if mesh_path is None or not mesh_path.exists():
        raise FileNotFoundError("Could not find a usable mesh file.")

    models_dir.mkdir(parents=True, exist_ok=True)
    mesh_dst = models_dir / "obj_000001.ply"
    if mesh_path.suffix.lower() == ".ply":
        shutil.copy2(mesh_path, mesh_dst)
    else:
        mesh = trimesh.load(mesh_path, force="mesh")
        mesh.export(mesh_dst)

    width, height = Image.open(rgb_files[0]).size
    cam_k = read_camera_matrix(source / "cam_K.txt")

    camera_json = {
        "cx": float(cam_k[0, 2]),
        "cy": float(cam_k[1, 2]),
        "fx": float(cam_k[0, 0]),
        "fy": float(cam_k[1, 1]),
        "width": int(width),
        "height": int(height),
        "depth_scale": 1.0,
    }
    (dataset_dir / "camera.json").write_text(json.dumps(camera_json, indent=2))
    (models_dir / "models_info.json").write_text(
        json.dumps(build_models_info(mesh_dst), indent=2)
    )

    scene_camera = {}
    scene_gt = {}
    scene_gt_info = {}
    test_targets = []

    for rgb_path in rgb_files:
        im_id = int(rgb_path.stem)
        bop_name = f"{im_id:06d}.png"

        copy_image(rgb_path, rgb_out / bop_name)
        if rgb_path.stem in depth_files:
            copy_image(depth_files[rgb_path.stem], depth_out / bop_name)
        if rgb_path.stem in mask_files:
            mask_name = f"{im_id:06d}_000000.png"
            copy_image(mask_files[rgb_path.stem], mask_out / mask_name)
            copy_image(mask_files[rgb_path.stem], mask_visib_out / mask_name)

        scene_camera[str(im_id)] = {
            "cam_K": cam_k.reshape(-1).tolist(),
            "depth_scale": 1.0,
        }
        scene_gt[str(im_id)] = []
        scene_gt_info[str(im_id)] = []
        test_targets.append(
            {
                "scene_id": 1,
                "im_id": im_id,
                "obj_id": 1,
                "inst_count": 1,
            }
        )

    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "scene_camera.json").write_text(json.dumps(scene_camera, indent=2))
    (scene_dir / "scene_gt.json").write_text(json.dumps(scene_gt, indent=2))
    (scene_dir / "scene_gt_info.json").write_text(json.dumps(scene_gt_info, indent=2))
    (dataset_dir / "test_targets_bop19.json").write_text(json.dumps(test_targets, indent=2))

    print(f"Wrote dataset to {dataset_dir}")
    print(f"Frames: {len(test_targets)}")
    print(f"Mesh: {mesh_dst}")


if __name__ == "__main__":
    main()
