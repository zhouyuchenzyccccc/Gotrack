from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import trimesh

from .data_prep import ensure_inference_results, ensure_multiview_bop_dataset, parse_camera_ids
from .depth_utils import DepthCache
from .io_utils import build_frame_paths, collect_available_frames, collect_frame_ids_from_raw, load_json, read_image
from .pose_filter import filter_large_pose_jump
from .pose_fusion import select_and_refine_multiview_pose, try_track_from_previous_pose
from .pose_io import load_pose_sequences
from .transforms import camera_pose_to_world_pose, split_transform
from .visualization import (
    Open3DSceneViewer,
    build_multiview_grid,
    build_single_view_grid,
    draw_status,
    handle_key,
    make_placeholder,
    render_fused_scene,
    update_playback_state,
    write_viewer_overlay,
)


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
            cv2.putText(wait, "Waiting for results...", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2, cv2.LINE_AA)
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
            collect_available_frames_fn=collect_available_frames,
        )

    cv2.destroyAllWindows()


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

    frame_counts: Dict[str, int] = {}
    result_dirs: Dict[str, Path] = {}
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

    pose_sequences = load_pose_sequences(result_dirs, frame_ids)

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(1.0 / 1000.0)
    mesh_points_m, _ = trimesh.sample.sample_surface(mesh, 2000)
    mesh_eval_points_m, _ = trimesh.sample.sample_surface(mesh, 700)
    overlay_dir = raw_data_dir / "gotrack_overlay"
    open3d_viewer = None
    if args.interactive_3d:
        open3d_viewer = Open3DSceneViewer(f"{args.window_name} 3D")

    delay_ms = max(1, int(1000 / max(args.fps, 0.1)))
    paused = False
    index = 0
    previous_accepted_pose_world: Optional[np.ndarray] = None
    previous_fused_confidence = 0.0
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    while True:
        frame_id = frame_ids[index]
        depth_cache = DepthCache(
            raw_data_dir=raw_data_dir,
            frame_id=frame_id,
            camera_ids=camera_ids,
            camera_params=camera_params,
            extrinsics=extrinsics,
            max_depth_m=args.max_depth_m,
        )
        depth_cache.preload()

        per_cam_world_poses = {}
        per_cam_scores = {}
        all_candidate_poses_world = {}
        fallback_cam_id = None
        fused_points = []
        for cam_id in camera_ids:
            pose = pose_sequences.get(cam_id, {}).get(frame_id)
            if pose is not None:
                per_cam_scores[cam_id] = pose["score"]
                pose_world = camera_pose_to_world_pose(pose, extrinsics[cam_id])
                all_candidate_poses_world[cam_id] = (pose_world, pose["score"])
                if pose["score"] >= args.min_pose_score:
                    per_cam_world_poses[cam_id] = (pose_world, pose["score"])
            fused_points.append(depth_cache.get_points_world(cam_id, args.point_stride))

        if not per_cam_world_poses and all_candidate_poses_world:
            fallback_cam_id = max(
                all_candidate_poses_world.keys(),
                key=lambda cam_id: all_candidate_poses_world[cam_id][1],
            )
            per_cam_world_poses = {
                fallback_cam_id: all_candidate_poses_world[fallback_cam_id]
            }

        track_result = None
        if args.track_from_previous:
            track_result = try_track_from_previous_pose(
                previous_pose_world=previous_accepted_pose_world,
                previous_confidence=previous_fused_confidence,
                camera_ids=camera_ids,
                depth_cache=depth_cache,
                camera_params=camera_params,
                extrinsics=extrinsics,
                mesh_points_m=mesh_eval_points_m,
                max_depth_m=args.max_depth_m,
                inlier_thresh_m=args.consistency_inlier_thresh_m,
                confidence_thresh=args.track_prev_confidence_thresh,
                consistency_thresh=args.track_prev_consistency_thresh,
            )

        if track_result is not None:
            fusion_result = track_result
            inlier_cam_ids = list(per_cam_world_poses.keys())
        else:
            fusion_result = select_and_refine_multiview_pose(
                candidate_poses_world=per_cam_world_poses,
                camera_ids=camera_ids,
                depth_cache=depth_cache,
                camera_params=camera_params,
                extrinsics=extrinsics,
                mesh_points_m=mesh_eval_points_m,
                max_depth_m=args.max_depth_m,
                inlier_thresh_m=args.consistency_inlier_thresh_m,
                view_consensus_thresh_m=args.view_consensus_thresh_m,
                view_consensus_min_views=args.view_consensus_min_views,
                candidate_cost_margin=args.candidate_cost_margin,
            )
            inlier_cam_ids = list(per_cam_world_poses.keys())

        fused_pose_world, jump_filter_status = filter_large_pose_jump(
            frame_id=frame_id,
            current_pose_world=fusion_result.pose_world,
            previous_pose_world=previous_accepted_pose_world,
            enabled=args.reject_large_jump,
            translation_thresh_m=args.reject_translation_jump_m,
            rotation_thresh_deg=args.reject_rotation_jump_deg,
        )
        status_parts = [part for part in [fusion_result.status, jump_filter_status] if part]
        filter_status = " | ".join(status_parts) if status_parts else None
        if fused_pose_world is not None:
            previous_accepted_pose_world = fused_pose_world.copy()
            if jump_filter_status is None:
                previous_fused_confidence = fusion_result.confidence

        points_world = np.concatenate(fused_points, axis=0) if fused_points else np.empty((0, 3), dtype=np.float32)
        object_points_world = np.empty((0, 3), dtype=np.float32)
        object_origin = None
        object_rotation = None
        if fused_pose_world is not None:
            object_rotation, object_origin = split_transform(fused_pose_world)
            object_points_world = ((object_rotation @ mesh_points_m.T).T + object_origin[None, :]).astype(np.float32)
        if args.export_viewer_overlay:
            write_viewer_overlay(overlay_dir=overlay_dir, frame_id=frame_id, object_points_world=object_points_world)

        candidate_costs = {
            cam_id: float(info["cost"])
            for cam_id, info in fusion_result.evaluations.items()
            if cam_id != "fused"
        }
        scene_image = render_fused_scene(
            points_world=points_world,
            fused_pose_world=fused_pose_world,
            mesh_points_m=mesh_points_m,
            title=f"Fused 3D pose | mode: {fusion_result.mode} | conf: {fusion_result.confidence:.3f}",
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
            selected_cam_ids=fusion_result.selected_cam_ids,
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
