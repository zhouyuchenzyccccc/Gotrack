from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import trimesh

from view_pose_pipeline.data_prep import ensure_multiview_bop_dataset, parse_camera_ids
from view_pose_pipeline.depth_utils import DepthCache
from view_pose_pipeline.io_utils import build_frame_paths, collect_frame_ids_from_raw, load_json, parse_pose_file, read_image
from view_pose_pipeline.pose_filter import filter_large_pose_jump
from view_pose_pipeline.pose_fusion import FusionResult, evaluate_pose_multiview, select_and_refine_multiview_pose
from view_pose_pipeline.transforms import camera_pose_to_world_pose, invert_transform, split_transform, world_from_rgb_extrinsic
from view_pose_pipeline.visualization import (
    Open3DSceneViewer,
    build_multiview_grid,
    draw_status,
    handle_key,
    make_placeholder,
    render_fused_scene,
    write_viewer_overlay,
)

from .model_loader import (
    build_camera_dataset_index,
    load_show3d_models,
    run_full_initialization_for_frame,
    run_refiner_only_for_frame,
)


@dataclass
class SequenceState:
    previous_pose_world: Optional[np.ndarray] = None
    previous_confidence: float = 0.0


def resolve_result_dir(repo_root: Path, dataset_name: str, results_root: Optional[Path]) -> Path:
    if results_root is None:
        return repo_root / "results" / "inference_pose_pipeline" / "localization" / dataset_name
    return results_root / dataset_name


def maybe_reuse_previous_pose(
    state: SequenceState,
    camera_ids: List[str],
    depth_cache: DepthCache,
    camera_params: Dict[str, Dict],
    extrinsics: Dict[str, Dict],
    mesh_points_m: np.ndarray,
    max_depth_m: float,
    inlier_thresh_m: float,
    reinit_confidence_thresh: float,
    reuse_consistency_thresh: float,
) -> Optional[FusionResult]:
    if state.previous_pose_world is None:
        return None
    if state.previous_confidence < reinit_confidence_thresh:
        return None
    evaluation = evaluate_pose_multiview(
        pose_world=state.previous_pose_world,
        camera_ids=camera_ids,
        depth_cache=depth_cache,
        camera_params=camera_params,
        extrinsics=extrinsics,
        mesh_points_m=mesh_points_m,
        max_depth_m=max_depth_m,
        inlier_thresh_m=inlier_thresh_m,
        source_score=state.previous_confidence,
    )
    if float(evaluation["consistency"]) < reuse_consistency_thresh:
        return None
    return FusionResult(
        pose_world=state.previous_pose_world.copy(),
        selected_cam_ids=[],
        evaluations={"prev_track": evaluation},
        confidence=float(evaluation["confidence"]),
        status=(
            "show3d-sequence: reuse previous pose"
            f" | conf={float(evaluation['confidence']):.3f}"
            f" | consistency={float(evaluation['consistency']):.3f}"
        ),
        mode="show3d_reuse_prev",
    )


def _unique_batch_idx(frame_id: int, cam_slot: int) -> int:
    return int(frame_id * 100 + cam_slot)


def _camera_pose_from_world_pose(world_pose: np.ndarray, extrinsics_entry: Dict) -> np.ndarray:
    t_world_cam = world_from_rgb_extrinsic(extrinsics_entry)
    t_cam_world = invert_transform(t_world_cam)
    return t_cam_world @ world_pose


def _load_cached_init_pose(result_dir: Path, frame_id: int):
    pose_path = result_dir / f"per_frame_refined_poses_{frame_id:06d}.json"
    return parse_pose_file(pose_path)


def run_show3d_sequence(args):
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
        raise ValueError("No valid camera ids found for SHOW3D sequence processing.")
    reference_camera = args.reference_camera if args.reference_camera in camera_ids else camera_ids[0]

    dataset_names: Dict[str, str] = {}
    camera_datasets = {}
    result_dirs: Dict[str, Path] = {}
    for cam_id in camera_ids:
        dataset_name, _ = ensure_multiview_bop_dataset(
            raw_data_dir=raw_data_dir,
            bop_root=bop_root,
            base_dataset_name=args.dataset_name,
            cam_id=cam_id,
            camera_params=camera_params,
            mesh_path=mesh_path,
            overwrite=args.overwrite_prepared,
        )
        dataset_names[cam_id] = dataset_name
        result_dirs[cam_id] = resolve_result_dir(repo_root, dataset_name, args.results_root)
        camera_datasets[cam_id] = build_camera_dataset_index(bop_root, dataset_name)

    canonical_dataset_name = dataset_names[camera_ids[0]]
    online_result_dir = repo_root / "results" / "show3d_pose_pipeline" / args.dataset_name
    online_result_dir.mkdir(parents=True, exist_ok=True)
    models = load_show3d_models(
        repo_root=repo_root,
        bop_root=bop_root,
        canonical_dataset_name=canonical_dataset_name,
        result_dir=online_result_dir,
        debug_vis=args.debug_vis,
    )

    frame_ids = collect_frame_ids_from_raw(raw_data_dir, camera_ids)
    if not frame_ids:
        raise FileNotFoundError("No synchronized frames found across the selected cameras.")

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(1.0 / 1000.0)
    mesh_points_m, _ = trimesh.sample.sample_surface(mesh, 2000)
    mesh_eval_points_m, _ = trimesh.sample.sample_surface(mesh, 700)

    state = SequenceState()
    overlay_dir = raw_data_dir / "show3d_overlay"
    open3d_viewer = Open3DSceneViewer(f"{args.window_name} 3D") if args.interactive_3d else None
    delay_ms = max(1, int(1000 / max(args.fps, 0.1)))
    paused = False
    index = 0
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
        fused_points = [depth_cache.get_points_world(cam_id, args.point_stride) for cam_id in camera_ids]
        fallback_cam_id = None

        previous_pose_reuse_probe = maybe_reuse_previous_pose(
            state=state,
            camera_ids=camera_ids,
            depth_cache=depth_cache,
            camera_params=camera_params,
            extrinsics=extrinsics,
            mesh_points_m=mesh_eval_points_m,
            max_depth_m=args.max_depth_m,
            inlier_thresh_m=args.consistency_inlier_thresh_m,
            reinit_confidence_thresh=args.reinit_confidence_thresh,
            reuse_consistency_thresh=args.reuse_consistency_thresh,
        )
        need_full_init = previous_pose_reuse_probe is None
        for cam_slot, cam_id in enumerate(camera_ids):
            dataset, frame_index = camera_datasets[cam_id]
            if frame_id not in frame_index:
                continue
            sample = dataset[frame_index[frame_id]]
            scene_observation = sample["scene_observation"]
            target_objects = sample["target_objects"]
            result = None
            if need_full_init and args.bootstrap_from_cache:
                cached_pose = _load_cached_init_pose(result_dirs[cam_id], frame_id)
                if cached_pose is not None:
                    pose_world = camera_pose_to_world_pose(cached_pose, extrinsics[cam_id])
                    result = {
                        "pose_cam_from_model": np.block(
                            [
                                [cached_pose["R"], cached_pose["t_m"].reshape(3, 1)],
                                [np.zeros((1, 3)), np.ones((1, 1))],
                            ]
                        ),
                        "score": float(cached_pose["score"]),
                        "pose_world": pose_world,
                    }
            if result is None and need_full_init:
                result = run_full_initialization_for_frame(
                    models=models,
                    scene_observation=scene_observation,
                    target_objects=target_objects,
                    batch_idx=_unique_batch_idx(frame_id, cam_slot),
                )
            elif result is None and state.previous_pose_world is not None:
                init_pose_cam = _camera_pose_from_world_pose(
                    state.previous_pose_world,
                    extrinsics[cam_id],
                )
                result = run_refiner_only_for_frame(
                    models=models,
                    scene_observation=scene_observation,
                    init_pose_cam_from_model=init_pose_cam,
                    obj_id=1,
                    batch_idx=_unique_batch_idx(frame_id, cam_slot),
                )

            if result is None:
                continue
            pose_cam = result["pose_cam_from_model"]
            pose_world = camera_pose_to_world_pose(
                {
                    "R": pose_cam[:3, :3],
                    "t_m": pose_cam[:3, 3],
                    "score": result["score"],
                },
                extrinsics[cam_id],
            )
            per_cam_scores[cam_id] = float(result["score"])
            per_cam_world_poses[cam_id] = (pose_world, float(result["score"]))

        if not per_cam_world_poses:
            fusion_result = (
                previous_pose_reuse_probe
                if previous_pose_reuse_probe is not None
                else FusionResult(None, [], {}, 0.0, status="no valid view", mode="show3d_online")
            )
        elif need_full_init:
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
            if fusion_result.status is None:
                fusion_result.status = "show3d-online: multiview reinit"
            if len(per_cam_world_poses) == 1:
                fallback_cam_id = next(iter(per_cam_world_poses.keys()))
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
            if fusion_result.status is None:
                fusion_result.status = "show3d-online: refine from previous pose"

        fused_pose_world, jump_filter_status = filter_large_pose_jump(
            frame_id=frame_id,
            current_pose_world=fusion_result.pose_world,
            previous_pose_world=state.previous_pose_world,
            enabled=args.reject_large_jump,
            translation_thresh_m=args.reject_translation_jump_m,
            rotation_thresh_deg=args.reject_rotation_jump_deg,
        )
        status_parts = [part for part in [fusion_result.status, jump_filter_status] if part]
        filter_status = " | ".join(status_parts) if status_parts else None
        if fused_pose_world is not None:
            state.previous_pose_world = fused_pose_world.copy()
            if jump_filter_status is None:
                state.previous_confidence = fusion_result.confidence

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
            title=f"SHOW3D-online pose | mode: {fusion_result.mode} | conf: {fusion_result.confidence:.3f}",
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
            inlier_cam_ids=list(per_cam_world_poses.keys()),
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
