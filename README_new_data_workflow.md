# GoTrack New Data Workflow

This guide is for running GoTrack on a new raw multi-camera sequence with
`scripts/view_pose_pipeline.py`.

It covers:

1. Environment setup
2. What files your raw data must contain
3. What to prepare once per object
4. What to run for each new sequence
5. Where outputs are written
6. Common failure cases

## 1. What `scripts/view_pose_pipeline.py` does

When you run [scripts/view_pose_pipeline.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/view_pose_pipeline.py),
the script does the following automatically for each selected camera:

1. Reads your raw multi-camera data under `--raw-data-dir`
2. Converts each camera stream into a minimal BOP dataset
3. Runs `scripts.inference_pose_estimation`
4. Reads each camera's per-frame refined pose
5. Fuses the poses across cameras
6. Displays the visualization window

So for a new sequence, you usually do not need to call separate data-prep and
inference scripts by hand.

## 2. One-time environment setup

Run these steps once on a new machine or a new environment.

### 2.1 Create the conda environment

```powershell
conda env create -f .\environment.yml
conda activate gotrack
bash .\scripts\env.sh
```

If `bash` is not available in your PowerShell environment, run the commands in
Git Bash or WSL instead:

```bash
conda env create -f environment.yml
conda activate gotrack
bash scripts/env.sh
```

### 2.2 Prepare checkpoints

Create a `checkpoints` directory under the repo root:

```powershell
New-Item -ItemType Directory -Force .\checkpoints
```

You need at least:

```text
Gotrack/
├─ checkpoints/
│  ├─ gotrack_checkpoint.pt
│  └─ FastSAM-x.pt
```

Notes:

- `gotrack_checkpoint.pt` is required by
  [scripts/view_pose_pipeline.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/view_pose_pipeline.py#L625)
  when it launches `scripts.inference_pose_estimation`.
- `FastSAM-x.pt` is referenced by
  [configs/model/cnos.yaml](/F:/论文下载/科研工程/gotrack-main/Gotrack/configs/model/cnos.yaml#L14)
  for the 2D detection stage.

## 3. Raw data directory you must provide

Your `--raw-data-dir` should look like this:

```text
your_sequence/
├─ camera_params.json
├─ extrinsics.json
├─ 00/
│  ├─ RGB/
│  │  ├─ 00000.jpg
│  │  ├─ 00001.jpg
│  │  └─ ...
│  └─ Depth/
│     ├─ 00000.png
│     ├─ 00001.png
│     └─ ...
├─ 01/
│  ├─ RGB/
│  └─ Depth/
├─ 02/
└─ ...
```

Important constraints:

- RGB file names must be integer frame ids such as `00000.jpg`, `00001.jpg`.
- Depth file names must match the RGB frame ids exactly, such as `00000.png`.
- Every selected camera must contain both `RGB/` and `Depth/`.
- The script uses the intersection of frame ids across cameras, so if one camera
  is missing frame `00037`, that frame will be skipped in fusion.

## 4. Required JSON fields

The script reads only a subset of fields, but those fields must exist.

### 4.1 `camera_params.json`

Minimum structure:

```json
{
  "00": {
    "RGB": {
      "intrinsic": {
        "fx": 0.0,
        "fy": 0.0,
        "cx": 0.0,
        "cy": 0.0
      }
    },
    "rgb_to_depth": {
      "depth_intrinsic": {
        "fx": 0.0,
        "fy": 0.0,
        "cx": 0.0,
        "cy": 0.0
      }
    }
  },
  "01": {}
}
```

This is required because the script reads:

- `camera_params[cam_id]["RGB"]["intrinsic"]`
- `camera_params[cam_id]["rgb_to_depth"]["depth_intrinsic"]`

See:

- [scripts/view_pose_pipeline.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/view_pose_pipeline.py#L486)
- [scripts/view_pose_pipeline.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/view_pose_pipeline.py#L823)

### 4.2 `extrinsics.json`

Minimum structure:

```json
{
  "00": {
    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "translation": [0.0, 0.0, 0.0],
    "rgb_to_depth": {
      "d2c_extrinsic": {
        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "translation": [0.0, 0.0, 0.0]
      }
    }
  }
}
```

You can also use `c2d_extrinsic`; the script supports both.

Unit conventions used by the current script:

- `extrinsics[cam_id]["translation"]` is used directly and should be in meters.
- `rgb_to_depth.d2c_extrinsic.translation` or
  `rgb_to_depth.c2d_extrinsic.translation` is divided by `1000.0` in code and
  should therefore be in millimeters.

See:

- [scripts/view_pose_pipeline.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/view_pose_pipeline.py#L676)
- [scripts/view_pose_pipeline.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/view_pose_pipeline.py#L685)

## 5. Mesh requirements

The mesh given by `--mesh-path` is used for:

- building the temporary BOP datasets
- template generation
- coarse pose estimation
- pose refinement
- 3D visualization

The current script assumes the mesh is in millimeters.

Why:

- GoTrack/BOP pipeline uses object translation values in millimeters
- the viewer later scales the mesh by `1.0 / 1000.0` for visualization

See:

- [scripts/view_pose_pipeline.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/view_pose_pipeline.py#L546)
- [scripts/view_pose_pipeline.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/view_pose_pipeline.py#L1243)

If your mesh is already in meters, the displayed fused pose will look wrong in
scale.

## 6. First run for a new object

If this object has never been run in this repo before, the first inference will
take longer because `scripts.inference_pose_estimation` will automatically do
object onboarding:

- render templates
- extract DINOv2 features
- build object representations

This is handled inside
[scripts/inference_pose_estimation.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/inference_pose_estimation.py#L55)
to
[scripts/inference_pose_estimation.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/inference_pose_estimation.py#L73).

You do not need to run a separate onboarding command if you are using
`scripts/view_pose_pipeline.py`.

## 7. Recommended end-to-end workflow for each new sequence

Assume:

- repo root: `F:\论文下载\科研工程\gotrack-main\Gotrack`
- raw sequence: `F:\data\drill_seq_01`
- temporary BOP root: `F:\data\bop_datasets`
- mesh: `F:\data\meshes\Scan.ply`
- cameras: `00,01,02,03,04,05`

### Step 1. Activate the environment

```powershell
cd F:\论文下载\科研工程\gotrack-main\Gotrack
conda activate gotrack
```

### Step 2. Verify the raw data directory

Check:

- `camera_params.json` exists
- `extrinsics.json` exists
- each selected camera has `RGB` and `Depth`
- frame ids match across cameras
- mesh path exists

### Step 3. Run the full pipeline

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

What happens in this run:

1. For each camera, the script creates `drill_00`, `drill_01`, ... under
   `F:\data\bop_datasets`
2. It runs GoTrack inference if results do not already exist
3. It opens the viewer window and fuses the multi-camera poses

### Step 4. Re-open only the visualization without rerunning inference

If per-camera results already exist, use `--skip-inference`:

```powershell
python .\scripts\view_pose_pipeline.py `
  --bop-root F:\data\bop_datasets `
  --dataset-name drill `
  --raw-data-dir F:\data\drill_seq_01 `
  --mesh-path F:\data\meshes\Scan.ply `
  --camera-ids 00,01,02,03,04,05 `
  --reference-camera 00 `
  --fps 5 `
  --skip-inference `
  --reject-translation-jump-m 0.05 `
  --reject-rotation-jump-deg 20
```

### Step 5. Force rebuilding the temporary BOP datasets

If you changed camera intrinsics, depth images, or mesh, add:

```powershell
--overwrite-prepared
```

Example:

```powershell
python .\scripts\view_pose_pipeline.py `
  --bop-root F:\data\bop_datasets `
  --dataset-name drill `
  --raw-data-dir F:\data\drill_seq_01 `
  --mesh-path F:\data\meshes\Scan.ply `
  --camera-ids 00,01,02,03,04,05 `
  --reference-camera 00 `
  --fps 5 `
  --overwrite-prepared
```

## 8. Where the outputs go

### 8.1 Prepared BOP datasets

For each camera:

```text
<bop-root>/<dataset-name>_<cam-id>/
```

Example:

```text
F:\data\bop_datasets\drill_00
F:\data\bop_datasets\drill_01
...
```

These are created by
[scripts/view_pose_pipeline.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/view_pose_pipeline.py#L507).

### 8.2 Per-camera inference results

```text
<repo-root>\results\inference_pose_pipeline\localization\<dataset-name>_<cam-id>\
```

Example:

```text
F:\论文下载\科研工程\gotrack-main\Gotrack\results\inference_pose_pipeline\localization\drill_00
```

Each directory should contain files such as:

```text
per_frame_refined_poses_000000.json
processed_detections_000000.png
vis_000000_foundPose.png
vis_000000_goTrack.png
```

The script checks these files in
[scripts/view_pose_pipeline.py](/F:/论文下载/科研工程/gotrack-main/Gotrack/scripts/view_pose_pipeline.py#L619).

### 8.3 Viewer overlay export

The fused object point cloud samples are exported to:

```text
<raw-data-dir>\gotrack_overlay\
```

Example:

```text
F:\data\drill_seq_01\gotrack_overlay
```

## 9. Useful options you will likely tune

### `--min-pose-score`

- Default: `0.6`
- Meaning: discard per-camera poses below this score before fusion

### `--reject-translation-jump-m`

- Unit: meters
- Meaning: if the fused pose translates too much relative to the previous
  accepted frame, replace it with the previous pose

### `--reject-rotation-jump-deg`

- Unit: degrees
- Meaning: if the fused pose rotates too much relative to the previous accepted
  frame, replace it with the previous pose

### `--skip-inference`

- Use this when per-camera inference has already been generated and you only want
  to view or tune the fusion/filter parameters

### `--interactive-3d`

- Opens an additional Open3D window
- Use only if Open3D is installed in your environment

## 10. Common failure cases

### Case 1. `checkpoints` directory does not exist

Symptom:

- inference fails before or during model loading

Fix:

- create `.\checkpoints`
- put `gotrack_checkpoint.pt` there
- put `FastSAM-x.pt` there

### Case 2. `camera_params.json` field mismatch

Symptom:

- KeyError on `RGB`, `intrinsic`, `rgb_to_depth`, or `depth_intrinsic`

Fix:

- rename your JSON keys to match the structure in Section 4

### Case 3. `extrinsics.json` unit mismatch

Symptom:

- fused object appears far away or in the wrong place
- projections are badly misaligned even when single-view results look plausible

Fix:

- ensure camera extrinsic translation is in meters
- ensure RGB-to-depth translation is in millimeters

### Case 4. Mesh unit mismatch

Symptom:

- object size in the fused 3D view is obviously wrong

Fix:

- use a mesh in millimeters

### Case 5. Existing prepared data is stale

Symptom:

- you changed raw images, intrinsics, or mesh, but the viewer still behaves like
  the old setup

Fix:

- rerun with `--overwrite-prepared`

### Case 6. Results exist but viewer shows missing frames

Symptom:

- some panels say "Waiting for result..." or there is no
  `per_frame_refined_poses_xxxxxx.json`

Fix:

- rerun without `--skip-inference`
- check whether that camera actually has the corresponding RGB and Depth frame

## 11. Minimal command set you can copy

### First full run on a new sequence

```powershell
cd F:\论文下载\科研工程\gotrack-main\Gotrack
conda activate gotrack
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

### Re-open the same sequence without rerunning inference

```powershell
cd F:\论文下载\科研工程\gotrack-main\Gotrack
conda activate gotrack
python .\scripts\view_pose_pipeline.py `
  --bop-root F:\data\bop_datasets `
  --dataset-name drill `
  --raw-data-dir F:\data\drill_seq_01 `
  --mesh-path F:\data\meshes\Scan.ply `
  --camera-ids 00,01,02,03,04,05 `
  --reference-camera 00 `
  --fps 5 `
  --skip-inference `
  --reject-translation-jump-m 0.05 `
  --reject-rotation-jump-deg 20
```

## 12. Suggested usage rhythm

For a completely new object:

1. Prepare the mesh in millimeters
2. Put the required checkpoints into `.\checkpoints`
3. Prepare one raw sequence in the folder structure above
4. Run the full pipeline once without `--skip-inference`
5. Re-open with `--skip-inference` and tune fusion/filter parameters

For another sequence of the same object:

1. Prepare the new raw sequence folder
2. Reuse the same mesh
3. Run the same command with a new `--raw-data-dir`
4. Add `--overwrite-prepared` only if the temporary BOP data should be rebuilt
