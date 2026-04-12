# view_pose_pipeline

`view_pose_pipeline` is the modular version of `scripts/view_pose_pipeline.py`.

It keeps the original end-to-end flow:

1. Prepare per-camera BOP datasets from raw multi-camera data
2. Run GoTrack pose estimation when needed
3. Fuse poses across cameras
4. Visualize the fused result

## File layout

| File | Responsibility |
|---|---|
| `config.py` | CLI arguments |
| `io_utils.py` | JSON/image/pose file I/O |
| `transforms.py` | SE(3) utilities |
| `data_prep.py` | BOP conversion and inference launching |
| `pose_io.py` | Per-camera pose sequence loading |
| `depth_utils.py` | Depth cache, world projection, depth reprojection |
| `pose_filter.py` | Large-jump filtering |
| `pose_fusion.py` | Multi-view fusion and SHOW3D-inspired tracking |
| `visualization.py` | OpenCV/Open3D/Matplotlib rendering |
| `viewer.py` | Single-view and multi-view playback loops |
| `run.py` | Package entrypoint |

## Entrypoints

Both commands work:

```powershell
python .\scripts\view_pose_pipeline.py ...
```

```powershell
python -m view_pose_pipeline.run ...
```

## Improvements over the old monolithic script

### Modular structure

You can now modify one part without touching the rest:

- raw-data to BOP conversion
- per-frame pose loading
- depth caching
- multi-view fusion
- temporal filtering
- visualization

### Faster fusion

The old script repeatedly loaded depth images from disk for every candidate pose.
The new pipeline adds `DepthCache` and preloads the current frame's depth maps
once, then reuses them for all candidate evaluations.

### More stable multi-view selection

The new fusion stage uses:

- view-consensus pruning to remove clearly inconsistent cameras
- multi-view depth consistency scoring for each candidate
- confidence-weighted transform averaging

### SHOW3D-inspired temporal fast path

SHOW3D Sec. 3.3 states that when the previous object pose remains reliable, the
pipeline should avoid expensive reinitialization and continue from the previous
pose, while using confidence to decide when to reinitialize.

This repo does not bundle PoseLib or a true multi-view gPnP implementation
inside FoundPose / GoTrack, so this package implements a practical fallback:

- if the previous fused pose had high confidence
- first validate that pose against the current frame's multi-view depth maps
- if it is still geometrically consistent, reuse it directly
- otherwise fall back to the full multi-view candidate selection

This follows the same control logic as SHOW3D, even though it is not a direct
copy of their internal multi-view gPnP solver.

## Usage

### Full end-to-end run

```powershell
python .\scripts\view_pose_pipeline.py `
  --bop-root F:\data\bop_datasets `
  --dataset-name drill `
  --raw-data-dir F:\data\drill_seq_01 `
  --mesh-path F:\data\meshes\Scan.ply `
  --camera-ids 00,01,02,03,04,05 `
  --reference-camera 00 `
  --fps 5 `
  --reject-translation-jump-m 0.05 `
  --reject-rotation-jump-deg 20
```

### Re-open results only

```powershell
python .\scripts\view_pose_pipeline.py `
  --bop-root F:\data\bop_datasets `
  --dataset-name drill `
  --raw-data-dir F:\data\drill_seq_01 `
  --mesh-path F:\data\meshes\Scan.ply `
  --camera-ids 00,01,02,03,04,05 `
  --reference-camera 00 `
  --fps 5 `
  --skip-inference
```

## New tuning options

### `--consistency-inlier-thresh-m`

- Depth reprojection residual threshold in meters
- Smaller values are stricter

### `--view-consensus-thresh-m`

- Translation threshold for removing outlier camera poses before fusion

### `--track-prev-confidence-thresh`

- Previous fused confidence required before the temporal fast path is allowed

### `--track-prev-consistency-thresh`

- Current-frame multi-view consistency required to keep tracking from the
  previous fused pose

## Recommended editing points

If you want to modify only one function family:

- Change raw-data to BOP conversion: `data_prep.py`
- Change cross-view pose selection: `pose_fusion.py`
- Change jump filtering: `pose_filter.py`
- Change overlays or panels: `visualization.py`
- Change execution order: `viewer.py`
