from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MESH_PATH = (
    "/home/ubuntu/WorkSpace/ZYC/FoundationPose/demo_data/electric_drill/mesh/Scan.ply"
)


def build_parser() -> argparse.ArgumentParser:
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
        help="Raw multi-camera directory, e.g. /path/to/raw/drill",
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
        help="Deprecated compatibility option kept for older commands.",
    )
    parser.add_argument(
        "--min-pose-score",
        type=float,
        default=0.6,
        help="Discard per-camera poses with score lower than this threshold.",
    )
    parser.add_argument(
        "--consistency-inlier-thresh-m",
        type=float,
        default=0.03,
        help="Depth reprojection residual threshold in meters when scoring pose consistency.",
    )
    parser.add_argument(
        "--view-consensus-thresh-m",
        type=float,
        default=0.08,
        help="Translation threshold used to remove outlier camera poses before fusion.",
    )
    parser.add_argument(
        "--view-consensus-min-views",
        type=int,
        default=2,
        help="Minimum number of views required for the consensus subset.",
    )
    parser.add_argument(
        "--candidate-cost-margin",
        type=float,
        default=0.015,
        help="Keep pose candidates whose multiview cost is within this margin from the best one.",
    )
    parser.add_argument(
        "--track-from-previous",
        action="store_true",
        default=True,
        help="SHOW3D-inspired fast path: validate the previous fused pose first and reuse it when still consistent.",
    )
    parser.add_argument(
        "--track-prev-confidence-thresh",
        type=float,
        default=0.55,
        help="Minimum previous fused confidence required to attempt tracking from the previous pose.",
    )
    parser.add_argument(
        "--track-prev-consistency-thresh",
        type=float,
        default=0.45,
        help="Minimum multiview consistency required to reuse the previous fused pose.",
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
    return parser


def parse_args():
    return build_parser().parse_args()
