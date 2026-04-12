"""Command-line argument parsing."""
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="GoTrack multi-view pose viewer (SHOW3D-inspired fusion)."
    )
    p.add_argument("--bop-root",        type=Path, required=True)
    p.add_argument("--dataset-name",    type=str,  required=True)
    p.add_argument("--results-dir",     type=Path, default=None)
    p.add_argument("--scene-id",        type=int,  default=1)
    p.add_argument("--fps",             type=float, default=8.0)
    p.add_argument("--watch",           action="store_true")
    p.add_argument("--window-name",     type=str,  default="GoTrack Pose Viewer")
    p.add_argument("--raw-data-dir",    type=Path, default=None)
    p.add_argument("--mesh-path",       type=Path, default=None)
    p.add_argument("--camera-ids",      type=str,  default="00,01,02,03,04,05")
    p.add_argument("--reference-camera",type=str,  default="00")
    p.add_argument("--skip-inference",  action="store_true")
    p.add_argument("--overwrite-prepared", action="store_true")
    p.add_argument("--max-depth-m",     type=float, default=2.0)
    p.add_argument("--point-stride",    type=int,   default=6)
    # Fusion
    p.add_argument("--ransac-thresh-m", type=float, default=0.08,
                   help="RANSAC translation consensus threshold (m).")
    p.add_argument("--depth-inlier-thresh-m", type=float, default=0.02,
                   help="Depth inlier threshold for multi-view consistency scoring (m).")
    p.add_argument("--min-pose-score",  type=float, default=0.6)
    # Temporal filter
    p.add_argument("--filter-mode",     choices=["hard", "ema"], default="ema",
                   help="Temporal filter: 'hard' (original) or 'ema' (smooth).")
    p.add_argument("--ema-alpha-t",     type=float, default=0.7,
                   help="EMA alpha for translation (0=frozen, 1=no smoothing).")
    p.add_argument("--ema-alpha-r",     type=float, default=0.6,
                   help="EMA alpha for rotation SLERP.")
    p.add_argument("--reject-translation-jump-m",  type=float, default=0.20)
    p.add_argument("--reject-rotation-jump-deg",   type=float, default=45.0)
    p.add_argument("--reject-large-jump", action="store_true", default=True)
    # Misc
    p.add_argument("--interactive-3d",  action="store_true")
    p.add_argument("--no-export-viewer-overlay", action="store_false",
                   dest="export_viewer_overlay")
    return p.parse_args()
