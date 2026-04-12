# view_pose_pipeline

Modular rewrite of `scripts/view_pose_pipeline.py` with SHOW3D-inspired multi-view pose fusion.

## Module layout

| File | Responsibility |
|---|---|
| `config.py` | CLI argument parsing |
| `transforms.py` | SE3 math (rotation, translation, inversion) |
| `depth_utils.py` | Depth loading with `DepthCache`, 3-D projection |
| `pose_fusion.py` | **SHOW3D-inspired** multi-view fusion |
| `pose_filter.py` | Temporal filtering: hard-reject or EMA |
| `data_prep.py` | BOP dataset preparation, inference runner |
| `visualization.py` | Matplotlib 3-D render, OpenCV panel/grid builders |
| `viewer.py` | Single-view and multi-view viewer loops |
| `run.py` | Entry point |

## Usage

```bash
# Multi-view (SHOW3D fusion, EMA smoothing)
python -m view_pose_pipeline.run \
  --bop-root /path/to/bop_datasets \
  --dataset-name electric_drill \
  --raw-data-dir /path/to/raw/drill \
  --mesh-path /path/to/Scan.ply \
  --camera-ids 00,01,02,03,04,05 \
  --reference-camera 00 \
  --fps 5 \
  --skip-inference

# Single-view (existing results only)
python -m view_pose_pipeline.run \
  --bop-root /path/to/bop_datasets \
  --dataset-name electric_drill
```

## Key improvements over the original script

### Speed
- **`DepthCache`** — depth images are read once per frame and shared across all
  scoring calls, eliminating redundant disk I/O.
- **Parallel depth preload** — all cameras' depth images for a frame are loaded
  concurrently via `ThreadPoolExecutor` before processing begins.

### Accuracy (SHOW3D-inspired)

The original script scored each camera's pose only against its own depth map,
then did a simple weighted average. The new pipeline follows the multi-view
consistency principle from SHOW3D (CVPR 2023):

1. **RANSAC view consensus** (`ransac_view_consensus`) — before fusing, find the
   largest subset of cameras whose translation estimates agree within
   `--ransac-thresh-m` (default 0.08 m). Outlier cameras are excluded.

2. **Multi-view depth consistency scoring** (`compute_multiview_consistency`) —
   each candidate pose is scored against **all** cameras' depth maps (not just
   the source camera). The score is the mean inlier ratio across views.

3. **Consistency-weighted fusion** — translation and rotation are averaged using
   weights = `detection_score × depth_consistency`, giving more influence to
   cameras that are both confident and geometrically consistent.

### Temporal smoothing

Two modes selectable via `--filter-mode`:

| Mode | Behaviour |
|---|---|
| `ema` (default) | Exponential moving average: `t_out = α·t_new + (1-α)·t_prev`, rotation via SLERP. Tunable with `--ema-alpha-t` and `--ema-alpha-r`. |
| `hard` | Original behaviour: reject frames where translation/rotation jump exceeds threshold and keep the previous pose. |

## Tuning guide

| Scenario | Recommendation |
|---|---|
| Noisy/fast motion | Lower `--ema-alpha-t` / `--ema-alpha-r` (more smoothing) |
| Cameras far apart | Increase `--ransac-thresh-m` |
| Depth sensor noise | Increase `--depth-inlier-thresh-m` |
| Low-confidence detections | Lower `--min-pose-score` |

## Keyboard controls

| Key | Action |
|---|---|
| `space` | Pause / resume |
| `a` / `d` | Step backward / forward one frame |
| `q` / `Esc` | Quit |

---

# view_pose_pipeline（中文说明）

对 `scripts/view_pose_pipeline.py` 的模块化重构，融合了 SHOW3D 论文的多视角位姿估计思路。

## 模块结构

| 文件 | 职责 |
|---|---|
| `config.py` | 命令行参数解析 |
| `transforms.py` | SE3 刚体变换数学工具（旋转、平移、求逆、四元数加权平均） |
| `depth_utils.py` | 深度图读取与缓存（`DepthCache`）、三维点投影 |
| `pose_fusion.py` | **SHOW3D 启发的**多视角位姿融合 |
| `pose_filter.py` | 时序滤波：硬拒绝（原始）或 EMA 平滑（新增） |
| `data_prep.py` | BOP 数据集准备、推理脚本调用 |
| `visualization.py` | Matplotlib 三维渲染、OpenCV 面板/网格拼接 |
| `viewer.py` | 单视角与多视角播放循环 |
| `run.py` | 程序入口 |

---

## 从头运行完整流程

### 原始数据目录结构

`--raw-data-dir` 下必须满足以下结构，否则流程无法启动：

```
drill/
├── camera_params.json        # 各相机内参（RGB intrinsic + Depth intrinsic）
├── extrinsics.json           # 各相机外参（rotation/translation + rgb_to_depth）
├── 00/
│   ├── RGB/    00000.jpg, 00001.jpg ...
│   └── Depth/  00000.png, 00001.png ...
├── 01/
│   ├── RGB/
│   └── Depth/
├── 02/ ...
```

---

### 第一步：下载模型权重（只需一次）

将 GoTrack 预训练权重放到：

```
Gotrack/checkpoints/gotrack_checkpoint.pt
```

---

### 第二步：Onboarding——为物体生成模板（每个新物体只需一次）

此步骤渲染物体的多视角模板图像并提取 DINOv2 特征，是检测和粗估计的基础。
**必须在 `Gotrack/` 目录下执行。**

```bash
cd Gotrack

# 2a. 生成渲染模板（约 57 个视角 × 14 个平面旋转）
python -m scripts.inference_gotrack \
  "onboarding.gen_templates_opts.dataset_name=electric_drill" \
  "onboarding.gen_templates_opts.object_lids=[1]" \
  "user.root_dir=$(pwd)"

# 2b. 提取特征表示（PCA 降维 + 聚类）
python -m scripts.inference_gotrack \
  "onboarding.gen_repre_opts.dataset_name=electric_drill" \
  "onboarding.gen_repre_opts.object_lids=[1]" \
  "user.root_dir=$(pwd)"
```

完成后会在 `results/` 下生成模板和特征文件，后续推理直接复用。

---

### 第三步：运行推理 + 多视角融合可视化

以下命令会自动完成：BOP 数据集准备 → 逐相机推理 → SHOW3D 多视角融合 → 实时可视化。

```bash
cd Gotrack

python -m view_pose_pipeline.run \
  --bop-root /path/to/bop_datasets \
  --dataset-name electric_drill \
  --raw-data-dir /path/to/raw/drill \
  --mesh-path /path/to/Scan.ply \
  --camera-ids 00,01,02,03,04,05 \
  --reference-camera 00 \
  --fps 5
```

**如果推理已经跑过，加 `--skip-inference` 直接看融合结果：**

```bash
python -m view_pose_pipeline.run \
  --bop-root /path/to/bop_datasets \
  --dataset-name electric_drill \
  --raw-data-dir /path/to/raw/drill \
  --mesh-path /path/to/Scan.ply \
  --camera-ids 00,01,02,03,04,05 \
  --reference-camera 00 \
  --fps 5 \
  --skip-inference
```

**单视角模式（无多相机原始数据，仅可视化已有推理结果）：**

```bash
python -m view_pose_pipeline.run \
  --bop-root /path/to/bop_datasets \
  --dataset-name electric_drill
```

---

### 各步骤是否自动

| 步骤 | 是否自动 | 备注 |
|---|---|---|
| 下载模型权重 | 手动 | 放到 `checkpoints/gotrack_checkpoint.pt` |
| Onboarding（模板生成 + 特征提取） | 手动，每个物体只做一次 | 见第二步 |
| BOP 数据集准备（原始数据 → BOP 格式） | **自动**（`view_pose_pipeline.run` 内部） | |
| 逐相机推理 | **自动**（除非加 `--skip-inference`） | |
| 多视角融合 + 可视化 | **自动** | |

## 相比原脚本的改进

### 速度提升

- **`DepthCache` 深度图缓存** — 每帧每个相机的深度图只从磁盘读取一次，后续所有评分调用共享缓存，消除重复 I/O。
- **并行深度图预加载** — 通过 `ThreadPoolExecutor` 在帧处理开始前并发加载所有相机的深度图。

### 精度提升（SHOW3D 启发）

原脚本仅将每个相机的位姿与自身深度图对比评分，再做简单加权平均。新流程遵循 SHOW3D（CVPR 2023）的多视角一致性原则：

1. **RANSAC 视角共识**（`ransac_view_consensus`）— 融合前，找出平移估计在 `--ransac-thresh-m`（默认 0.08 m）范围内互相一致的最大相机子集，直接排除离群相机。

2. **多视角深度一致性评分**（`compute_multiview_consistency`）— 每个候选位姿都与**所有**相机的深度图对比（而非仅与来源相机对比），得分为跨视角的平均内点率。这是 SHOW3D 的核心思想：正确的位姿在每个视角的重投影都应一致。

3. **一致性加权融合** — 融合权重 = `检测置信度 × 深度一致性`，几何上更可靠的相机对最终结果贡献更大。

### 时序平滑

通过 `--filter-mode` 选择：

| 模式 | 行为 |
|---|---|
| `ema`（默认） | 指数移动平均：平移 `t_out = α·t_new + (1-α)·t_prev`，旋转用 SLERP 插值。通过 `--ema-alpha-t` 和 `--ema-alpha-r` 调节。 |
| `hard` | 原始行为：当平移/旋转跳变超过阈值时拒绝当前帧，保留上一帧位姿。 |

## 参数调节指南

| 场景 | 建议 |
|---|---|
| 运动噪声大 / 物体移动快 | 降低 `--ema-alpha-t` / `--ema-alpha-r`（增强平滑） |
| 相机间距较大 | 增大 `--ransac-thresh-m` |
| 深度传感器噪声大 | 增大 `--depth-inlier-thresh-m` |
| 检测置信度普遍偏低 | 降低 `--min-pose-score` |

## 键盘操作

| 按键 | 功能 |
|---|---|
| `空格` | 暂停 / 继续播放 |
| `a` / `d` | 后退 / 前进一帧 |
| `q` / `Esc` | 退出 |
