# show3d_pose_pipeline

This folder contains a SHOW3D-oriented object pose pipeline entrypoint.

The goal of this package is to follow the object-pose method described in
SHOW3D as closely as the public materials allow.

## What is explicitly stated in SHOW3D

From SHOW3D Sec. 3.3:

- the object pipeline is `CNOS -> FoundPose -> GoTrack`
- the detector runs on all available views
- the highest-confidence detection in each view is used
- standard PnP is replaced by multi-view gPnP from PoseLib
- the first two stages run only in the first frame, or when the previous-frame
  confidence is too low
- otherwise, only the pose refiner is executed using the previous pose as the
  initial pose

## Important limitation

There is currently no public official SHOW3D code release linked from the
paper page, and the paper does not publish the exact implementation details
needed for a bit-for-bit reproduction of their internal object-pose stack.

Examples of missing public details:

- the exact multi-view modifications inside their FoundPose implementation
- the exact multi-view modifications inside their GoTrack refiner
- the exact confidence threshold used for reinitialization
- the exact PoseLib RANSAC and bundle-adjustment options
- any unpublished implementation-specific preprocessing

Because of that, this package is a paper-grounded reconstruction, not a
guaranteed byte-identical reproduction of the unpublished internal code.

## What is implemented here

### 1. SHOW3D-style sequence control

- if the previous fused pose is still confident and geometrically consistent,
  reuse it
- otherwise, reinitialize from the current multi-view candidates

This mirrors the control policy described in SHOW3D Sec. 3.3.

### 2. PoseLib interface layer

`poselib_utils.py` adds wrappers for:

- generalized absolute pose estimation
- generalized absolute pose refinement

This is the correct solver family for the paper's reported multi-view gPnP
direction, and can be wired into deeper FoundPose / GoTrack internals later.

### 3. Current runnable path

The current runnable entrypoint is an online sequence pipeline:

- first frame or low-confidence frame:
  run `CNOS -> FoundPose -> GoTrack` per view
- otherwise:
  skip detection and coarse pose estimation
- project the previous fused world pose into each camera
- run only GoTrack refinement per view
- fuse the refined per-view poses across cameras

This is the main practical acceleration path implemented here.

### 4. Optional cache bootstrap

If you already have cached per-camera `per_frame_refined_poses_*.json` files,
you can enable:

- `--bootstrap-from-cache`

This uses the cached pose on reinitialization frames instead of running a fresh
online initialization.

## Remote environment note

Per your request, PoseLib has already been installed on the remote host:

- host: `ubuntu@10.162.241.5`
- path: `/home/ubuntu/WorkSpace/ZYC/Gotrack`
- conda env: `gotrack`

Installed package:

- `poselib`

That installation is environment-side only and is not part of git-tracked code.

## Usage

```powershell
python -m show3d_pose_pipeline.run `
  --bop-root F:\data\bop_datasets `
  --dataset-name drill `
  --raw-data-dir F:\data\drill_seq_01 `
  --mesh-path F:\data\meshes\Scan.ply `
  --camera-ids 00,01,02,03,04,05 `
  --reference-camera 00 `
  --fps 5 `
  --skip-inference
```

### Cache-assisted bootstrap

```powershell
python -m show3d_pose_pipeline.run `
  --bop-root F:\data\bop_datasets `
  --dataset-name drill `
  --raw-data-dir F:\data\drill_seq_01 `
  --mesh-path F:\data\meshes\Scan.ply `
  --camera-ids 00,01,02,03,04,05 `
  --reference-camera 00 `
  --bootstrap-from-cache `
  --skip-inference
```

## Tuning options

### `--reinit-confidence-thresh`

- if the previous fused confidence is below this, rerun initialization

### `--reuse-consistency-thresh`

- if the previous fused pose no longer matches the current multi-view depth
  evidence, rerun initialization

### `--view-consensus-thresh-m`

- remove obviously inconsistent camera poses before multiview selection

## Files

| File | Role |
|---|---|
| `config.py` | CLI arguments |
| `poselib_utils.py` | PoseLib generalized PnP wrappers |
| `model_loader.py` | model construction, onboarding, init/refiner helpers |
| `controller.py` | SHOW3D-style online sequence controller |
| `run.py` | entrypoint |
