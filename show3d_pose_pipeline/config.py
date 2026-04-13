from __future__ import annotations

import argparse
from pathlib import Path

from view_pose_pipeline.config import DEFAULT_MESH_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SHOW3D-style multiview object pose pipeline built on top of the current GoTrack repo."
    )
    parser.add_argument("--bop-root", type=Path, required=True, help="Path to bop_datasets.")
    parser.add_argument("--dataset-name", type=str, required=True, help="Base dataset name.")
    parser.add_argument("--raw-data-dir", type=Path, required=True, help="Raw multi-camera directory.")
    parser.add_argument("--mesh-path", type=Path, default=Path(DEFAULT_MESH_PATH), help="Object mesh path.")
    parser.add_argument("--camera-ids", type=str, default="00,01,02,03,04,05", help="Comma-separated camera ids.")
    parser.add_argument("--reference-camera", type=str, default="00", help="Reference camera for visualization.")
    parser.add_argument("--results-root", type=Path, default=None, help="Optional override for per-camera result root.")
    parser.add_argument(
        "--bootstrap-from-cache",
        action="store_true",
        help="Use cached per-camera refined poses for reinitialization if available instead of running full initialization online.",
    )
    parser.add_argument(
        "--debug-vis",
        action="store_true",
        help="Enable debug visualizations for CNOS / FoundPose / GoTrack.",
    )
    parser.add_argument("--fps", type=float, default=5.0, help="Playback FPS.")
    parser.add_argument("--skip-inference", action="store_true", help="Reuse existing per-camera results.")
    parser.add_argument("--overwrite-prepared", action="store_true", help="Rebuild prepared BOP datasets.")
    parser.add_argument("--min-pose-score", type=float, default=0.6, help="Minimum per-view refined pose score.")
    parser.add_argument("--max-depth-m", type=float, default=2.0, help="Maximum depth in meters.")
    parser.add_argument("--point-stride", type=int, default=6, help="Stride for fused depth visualization.")
    parser.add_argument(
        "--consistency-inlier-thresh-m",
        type=float,
        default=0.03,
        help="Depth reprojection residual threshold in meters when scoring pose consistency.",
    )
    parser.add_argument(
        "--reinit-confidence-thresh",
        type=float,
        default=0.55,
        help="If previous fused confidence falls below this threshold, rerun full initialization.",
    )
    parser.add_argument(
        "--reuse-consistency-thresh",
        type=float,
        default=0.45,
        help="If the previous fused pose still has this multiview consistency, skip reinitialization.",
    )
    parser.add_argument(
        "--view-consensus-thresh-m",
        type=float,
        default=0.08,
        help="Translation threshold for camera-view consensus pruning.",
    )
    parser.add_argument(
        "--view-consensus-min-views",
        type=int,
        default=2,
        help="Minimum views for consensus pruning to take effect.",
    )
    parser.add_argument(
        "--candidate-cost-margin",
        type=float,
        default=0.015,
        help="Keep candidates whose multiview cost is within this margin from the best one.",
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
    parser.add_argument("--window-name", type=str, default="SHOW3D Pose Viewer")
    parser.add_argument("--interactive-3d", action="store_true", help="Open an Open3D viewer.")
    parser.add_argument(
        "--no-export-viewer-overlay",
        action="store_false",
        dest="export_viewer_overlay",
        help="Disable exporting fused mesh samples for the sync viewer.",
    )
    return parser


def parse_args():
    return build_parser().parse_args()
