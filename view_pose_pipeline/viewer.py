"""Single-view and multi-view viewer loops."""
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh

from .data_prep import ensure_inference_results, ensure_multiview_bop_dataset
from .depth_utils import DepthCache, load_depth_points_world
from .pose_filter import EMAFilter, filter_large_pose_jump
from .pose_fusion import show3d_fuse_poses
from .transforms import camera_pose_to_world_pose, split_transform
from .visualization import (
    add_title,
    build_multiview_grid,
    build_single_view_grid,
    draw_status,
    make_placeholder,
    read_image,
    render_fused_scene,
)

try:
    import open3d as o3d
except Exception:
    o3d = None


# ---------------------------------------------------------------------------
# Pose file I/O
# ---------------------------------------------------------------------------

def parse_pose_file(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    row = rows[0]
    return {
        "R":   np.fromstring(row["R"], sep=" ", dtype=np.float64).reshape(3, 3),
        "t_m": np.fromstring(row["t"], sep=" ", dtype=np.float64) / 1000.0,
        "score": float(row["score"]),
    }


def build_frame_paths(results_dir: Path, frame_id: int) -> Dict[str, Path]:
    return {
        "detection": results_dir / f"processed_detections_{frame_id:06d}.png",
        "coarse":    results_dir / f"vis_{frame_id:06d}_foundPose.png",
        "refined":   results_dir / f"vis_{frame_id:06d}_goTrack.png",
    }


# ---------------------------------------------------------------------------
# Viewer overlay export
# ---------------------------------------------------------------------------

def write_viewer_overlay(overlay_dir: Path, frame_id: int, object_points_world: np.ndarray):
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out = overlay_dir / f"frame_{frame_id:05d}.xyzrgb"
    if object_points_world.size == 0:
        if out.exists():
            out.unlink()
        return
    with out.open("w", encoding="utf-8") as fh:
        for p in object_points_world:
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 255 60 60\n")


# ---------------------------------------------------------------------------
# Frame collection helpers
# ---------------------------------------------------------------------------

def collect_available_frames(results_dir: Path) -> List[int]:
    ids = set()
    for pat in ["processed_detections_*.png", "vis_*_foundPose.png", "vis_*_goTrack.png"]:
        for p in results_dir.glob(pat):
            for part in p.stem.split("_"):
                if part.isdigit():
                    ids.add(int(part))
                    break
    return sorted(ids)


def collect_frame_ids_from_raw(raw_data_dir: Path, camera_ids: List[str]) -> List[int]:
    sets = [{int(p.stem) for p in (raw_data_dir / c / "RGB").glob("*.jpg")} for c in camera_ids]
    return sorted(set.intersection(*sets)) if sets else []


def parse_camera_ids(arg: str, raw_data_dir: Path) -> List[str]:
    requested = [s.strip() for s in arg.split(",") if s.strip()]
    available = {p.name for p in raw_data_dir.iterdir() if p.is_dir()}
    ids = [c for c in requested if c in available]
    if not ids:
        ids = sorted([c for c in available if c.isdigit()])[:6]
    return ids


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------

def handle_key(key: int) -> bool:
    return key in (ord("q"), 27)


# ---------------------------------------------------------------------------
# Single-view viewer
# ---------------------------------------------------------------------------

def run_single_view_viewer(args):
    results_dir = (
        args.results_dir.resolve() if args.results_dir
        else (args.bop_root.parent / "results" / "inference_pose_pipeline"
              / "localization" / args.dataset_name).resolve()
    )
    rgb_dir = (
        args.bop_root.resolve() / args.dataset_name / "test"
        / f"{args.scene_id:06d}" / "rgb"
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
                raise FileNotFoundError(f"No result frames in {results_dir}")
            wait = np.full((360, 720, 3), 250, dtype=np.uint8)
            cv2.putText(wait, "Waiting for results...", (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2, cv2.LINE_AA)
            cv2.imshow(args.window_name, wait)
            if cv2.waitKey(500) & 0xFF in (ord("q"), 27):
                break
            frame_ids = collect_available_frames(results_dir)
            continue

        frame_id = frame_ids[index]
        rgb = read_image(rgb_dir / f"{frame_id:06d}.png") or make_placeholder(
            (480, 848, 3), "RGB", f"Missing frame {frame_id:06d}"
        )
        grid = build_single_view_grid(rgb, build_frame_paths(results_dir, frame_id), frame_id)
        draw_status(grid, paused)
        cv2.imshow(args.window_name, grid)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if handle_key(key):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("a"):
            index, paused = max(0, index - 1), True
        elif key == ord("d"):
            index, paused = min(len(frame_ids) - 1, index + 1), True
        elif not paused:
            frame_ids = collect_available_frames(results_dir) or frame_ids
            if index < len(frame_ids) - 1:
                index += 1
            elif args.watch:
                time.sleep(0.1)

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Multi-view viewer (SHOW3D-inspired fusion)
# ---------------------------------------------------------------------------

def run_multiview_viewer(args):
    raw_data_dir = args.raw_data_dir.resolve()
    repo_root = Path(__file__).resolve().parent.parent
    bop_root = args.bop_root.resolve()
    mesh_path = args.mesh_path.resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    camera_params = json.loads((raw_data_dir / "camera_params.json").read_text())
    extrinsics = json.loads((raw_data_dir / "extrinsics.json").read_text())
    camera_ids = [
        c for c in parse_camera_ids(args.camera_ids, raw_data_dir) if c in extrinsics
    ]
    if not camera_ids:
        raise ValueError("No valid camera ids found.")
    ref_cam = args.reference_camera if args.reference_camera in camera_ids else camera_ids[0]

    # Prepare BOP datasets and run inference
    result_dirs: Dict[str, Path] = {}
    for cam_id in camera_ids:
        ds_name, fc = ensure_multiview_bop_dataset(
            raw_data_dir, bop_root, args.dataset_name, cam_id,
            camera_params, mesh_path, args.overwrite_prepared,
        )
        result_dirs[cam_id] = ensure_inference_results(
            repo_root, ds_name, fc, args.skip_inference,
        )

    frame_ids = collect_frame_ids_from_raw(raw_data_dir, camera_ids)
    if not frame_ids:
        raise FileNotFoundError("No synchronized frames found.")

    # Load mesh for visualisation and evaluation
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(1.0 / 1000.0)
    mesh_vis_pts, _ = trimesh.sample.sample_surface(mesh, 2500)
    mesh_eval_pts, _ = trimesh.sample.sample_surface(mesh, 900)

    overlay_dir = raw_data_dir / "gotrack_overlay"
    depth_cache = DepthCache(max_entries=len(camera_ids) * 6)

    # Temporal filter
    ema_filter = EMAFilter(alpha_t=args.ema_alpha_t, alpha_r=args.ema_alpha_r)
    previous_accepted_pose: Optional[np.ndarray] = None

    delay_ms = max(1, int(1000 / max(args.fps, 0.1)))
    paused = False
    index = 0
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    while True:
        frame_id = frame_ids[index]

        # Preload all depth images for this frame in parallel
        depth_cache.preload_frame(frame_id, camera_ids, raw_data_dir)

        # Collect per-camera poses
        per_cam_scores: Dict[str, float] = {}
        candidate_poses: Dict[str, Tuple[np.ndarray, float]] = {}
        for cam_id in camera_ids:
            pose_path = result_dirs[cam_id] / f"per_frame_refined_poses_{frame_id:06d}.json"
            pose = parse_pose_file(pose_path)
            if pose is not None:
                per_cam_scores[cam_id] = pose["score"]
                if pose["score"] >= args.min_pose_score:
                    candidate_poses[cam_id] = (
                        camera_pose_to_world_pose(pose, extrinsics[cam_id]),
                        pose["score"],
                    )

        # Fallback: use best-score camera even if below threshold
        if not candidate_poses and per_cam_scores:
            best = max(per_cam_scores, key=per_cam_scores.get)
            pose = parse_pose_file(result_dirs[best] / f"per_frame_refined_poses_{frame_id:06d}.json")
            if pose is not None:
                candidate_poses[best] = (
                    camera_pose_to_world_pose(pose, extrinsics[best]),
                    pose["score"],
                )

        # SHOW3D-inspired fusion
        fused_pose, consensus_ids, consistency_scores = show3d_fuse_poses(
            candidate_poses_world=candidate_poses,
            frame_id=frame_id,
            camera_ids=camera_ids,
            raw_data_dir=raw_data_dir,
            camera_params=camera_params,
            extrinsics=extrinsics,
            mesh_points_m=mesh_eval_pts,
            max_depth_m=args.max_depth_m,
            depth_cache=depth_cache,
            ransac_thresh_m=args.ransac_thresh_m,
            inlier_thresh_m=args.depth_inlier_thresh_m,
        )

        # Temporal filtering
        filter_status: Optional[str] = None
        if fused_pose is not None:
            if args.filter_mode == "ema":
                fused_pose = ema_filter.update(fused_pose)
            else:
                fused_pose, filter_status = filter_large_pose_jump(
                    frame_id, fused_pose, previous_accepted_pose,
                    args.reject_large_jump,
                    args.reject_translation_jump_m,
                    args.reject_rotation_jump_deg,
                )
        if fused_pose is not None:
            previous_accepted_pose = fused_pose.copy()

        # Build point clouds
        fused_pts = np.concatenate([
            load_depth_points_world(
                raw_data_dir, c, frame_id, camera_params, extrinsics,
                args.point_stride, args.max_depth_m, depth_cache,
            ) for c in camera_ids
        ], axis=0)

        obj_pts_world = np.empty((0, 3), dtype=np.float32)
        if fused_pose is not None:
            R, t = split_transform(fused_pose)
            obj_pts_world = ((R @ mesh_vis_pts.T).T + t[None, :]).astype(np.float32)

        if args.export_viewer_overlay:
            write_viewer_overlay(overlay_dir, frame_id, obj_pts_world)

        # Render
        filtered_label = (
            f"Fused 3D | views: {len(consensus_ids)}"
            if filter_status is None
            else f"Fused 3D | views: {len(consensus_ids)} | filtered"
        )
        scene_img = render_fused_scene(fused_pts, fused_pose, mesh_vis_pts, filtered_label)

        ref_rgb = read_image(raw_data_dir / ref_cam / "RGB" / f"{frame_id:05d}.jpg")
        if ref_rgb is None:
            ref_rgb = make_placeholder((480, 848, 3), "RGB", "Missing reference RGB")

        grid = build_multiview_grid(
            raw_rgb=ref_rgb,
            frame_paths=build_frame_paths(result_dirs[ref_cam], frame_id),
            frame_id=frame_id,
            fused_scene=scene_img,
            consensus_cam_ids=consensus_ids,
            per_cam_scores=per_cam_scores,
            consistency_scores=consistency_scores,
            min_pose_score=args.min_pose_score,
            filter_status=filter_status,
        )
        draw_status(grid, paused)
        cv2.imshow(args.window_name, grid)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if handle_key(key):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("a"):
            index, paused = max(0, index - 1), True
            ema_filter.reset()
        elif key == ord("d"):
            index, paused = min(len(frame_ids) - 1, index + 1), True
        elif not paused and index < len(frame_ids) - 1:
            index += 1

    cv2.destroyAllWindows()
