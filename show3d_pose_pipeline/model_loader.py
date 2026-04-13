from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dataloader.bop import BopDataset
from utils import data_util, gen_repre_util, net_util, structs, template_util


@dataclass
class Show3DModels:
    detector: object
    coarse_pose_model: object
    refiner_model: object
    canonical_dataset: BopDataset
    repre_dir: Path


def _load_yaml(path: Path):
    return OmegaConf.load(str(path))


def _prepare_object_onboarding(
    bop_root: Path,
    dataset_name: str,
    object_ids: List[int],
    onboarding_cfg,
) -> Path:
    onboarding_cfg = OmegaConf.create(OmegaConf.to_container(onboarding_cfg, resolve=True))
    onboarding_cfg.gen_templates_opts.dataset_name = dataset_name
    onboarding_cfg.gen_templates_opts.object_lids = object_ids
    onboarding_cfg.gen_templates_opts.overwrite = False
    onboarding_cfg.gen_repre_opts.dataset_name = dataset_name
    onboarding_cfg.gen_repre_opts.object_lids = object_ids
    template_util.batch_synthesize_templates(
        bop_root_dir=bop_root,
        opts=instantiate(onboarding_cfg.gen_templates_opts),
        num_workers=1,
    )
    return gen_repre_util.generate_repre_from_list(
        bop_root_dir=bop_root,
        opts=instantiate(onboarding_cfg.gen_repre_opts),
    )


def load_show3d_models(
    repo_root: Path,
    bop_root: Path,
    canonical_dataset_name: str,
    result_dir: Path,
    debug_vis: bool,
) -> Show3DModels:
    configs_dir = repo_root / "configs" / "model"
    all_cfg = _load_yaml(configs_dir / "all.yaml")
    cnos_cfg = _load_yaml(configs_dir / "cnos.yaml")
    foundpose_cfg = _load_yaml(configs_dir / "foundpose.yaml")
    gotrack_cfg = _load_yaml(configs_dir / "gotrack.yaml")
    onboarding_cfg = _load_yaml(configs_dir / "onboarding.yaml")

    common_crop_size = tuple(all_cfg.crop_size)
    common_crop_rel_pad = float(all_cfg.crop_rel_pad)
    cnos_cfg.opts.crop_size = common_crop_size
    cnos_cfg.opts.crop_rel_pad = common_crop_rel_pad
    cnos_cfg.opts.debug = debug_vis
    cnos_cfg.segmentation_model.opts.model_path = str(repo_root / "checkpoints" / "FastSAM-x.pt")

    foundpose_cfg.opts.crop_size = common_crop_size
    foundpose_cfg.opts.crop_rel_pad = common_crop_rel_pad
    foundpose_cfg.opts.debug = debug_vis

    gotrack_cfg.opts.crop_size = common_crop_size
    gotrack_cfg.opts.crop_rel_pad = common_crop_rel_pad
    gotrack_cfg.opts.debug = debug_vis
    gotrack_cfg.checkpoint_path = str(repo_root / "checkpoints" / "gotrack_checkpoint.pt")

    canonical_dataset = BopDataset(
        root_dir=str(bop_root),
        dataset_name=canonical_dataset_name,
        is_localization_task=True,
        use_default_detections=False,
        coarse_pose_method=None,
    )
    object_ids = sorted(list(canonical_dataset.models_info.keys()))
    repre_dir = _prepare_object_onboarding(
        bop_root=bop_root,
        dataset_name=canonical_dataset_name,
        object_ids=object_ids,
        onboarding_cfg=onboarding_cfg,
    )

    detector = instantiate(cnos_cfg, result_dir=result_dir, result_file_name="show3d_cnos")
    coarse_pose_model = instantiate(
        foundpose_cfg,
        result_dir=result_dir,
        result_file_name="show3d_foundpose",
    )
    refiner_model = instantiate(
        gotrack_cfg,
        result_dir=result_dir,
        result_file_name="show3d_gotrack",
    )

    detector = detector.cuda().eval()
    coarse_pose_model = coarse_pose_model.cuda().eval()
    refiner_model = refiner_model.cuda().eval()
    coarse_pose_model.set_renderer(canonical_dataset)
    refiner_model.set_renderer(canonical_dataset)

    net_util.load_checkpoint(
        model=refiner_model,
        checkpoint_path=gotrack_cfg.checkpoint_path,
        checkpoint_key="model_state_dict",
        prefix="models.1.",
    )

    detector.objects_repre = detector.onboarding(repre_dir)
    detector.post_onboarding_processing()
    coarse_pose_model.objects_repre = coarse_pose_model.onboarding(repre_dir)
    coarse_pose_model.post_onboarding_processing()
    refiner_model.objects_repre = refiner_model.onboarding(repre_dir)

    return Show3DModels(
        detector=detector,
        coarse_pose_model=coarse_pose_model,
        refiner_model=refiner_model,
        canonical_dataset=canonical_dataset,
        repre_dir=repre_dir,
    )


def build_camera_dataset_index(
    bop_root: Path,
    dataset_name: str,
) -> tuple[BopDataset, Dict[int, int]]:
    dataset = BopDataset(
        root_dir=str(bop_root),
        dataset_name=dataset_name,
        is_localization_task=True,
        use_default_detections=False,
        coarse_pose_method=None,
    )
    frame_index = {}
    for idx, key in enumerate(dataset.targets_per_image.keys()):
        _, im_id = key
        frame_index[int(im_id)] = idx
    return dataset, frame_index


def run_full_initialization_for_frame(
    models: Show3DModels,
    scene_observation,
    target_objects,
    batch_idx: int,
):
    detection_results = models.detector.forward_pipeline(
        scene_observation=scene_observation,
        batch_idx=batch_idx,
    )
    if target_objects is not None:
        detection_results = models.detector.filter_detections_by_targets(
            detection_results,
            target_objects=target_objects,
        )
    coarse_pose_results = models.coarse_pose_model.forward_pipeline(
        inputs=detection_results,
        batch_idx=batch_idx,
    )
    refiner_inputs = data_util.convert_foundpose_outputs_to_gotrack_inputs(coarse_pose_results)
    refiner_outputs = models.refiner_model.forward_pipeline(
        inputs=refiner_inputs,
        batch_idx=batch_idx,
    )
    objects = refiner_outputs["objects"]
    if len(objects) == 0:
        return None
    best_idx = int(torch.argmax(objects.pose_scores).item())
    return {
        "pose_cam_from_model": objects.poses_cam_from_model[best_idx].detach().cpu().numpy(),
        "score": float(objects.pose_scores[best_idx].detach().cpu().item()),
    }


def run_refiner_only_for_frame(
    models: Show3DModels,
    scene_observation,
    init_pose_cam_from_model,
    obj_id: int,
    batch_idx: int,
):
    pose = init_pose_cam_from_model.astype("float32")
    anno = structs.ObjectAnnotation(
        dataset=models.canonical_dataset.dataset_name,
        lid=int(obj_id),
        pose=structs.ObjectPose(
            R=pose[:3, :3],
            t=pose[:3, 3:].reshape(3, 1),
        ),
    )
    scene_observation = scene_observation._replace(objects_anno=[anno], time=0.0)
    inputs = data_util.convert_list_scene_observations_to_gotrack_inputs(
        [{"scene_observation": scene_observation}]
    )
    processed = data_util.process_inputs(
        inputs=inputs,
        crop_size=models.refiner_model.opts.crop_size,
        crop_rel_pad=models.refiner_model.opts.crop_rel_pad,
        cropping_type=models.refiner_model.opts.cropping_type,
        ssaa_factor=models.refiner_model.opts.ssaa_factor,
        object_vertices=[models.refiner_model.object_vertices[int(obj_id)]],
        obj_ids=[int(obj_id)],
        renderer=models.refiner_model.renderer,
        background_type=models.refiner_model.opts.background_type,
    )
    outputs = models.refiner_model.forward_pipeline(processed, batch_idx=batch_idx)
    objects = outputs["objects"]
    if len(objects) == 0:
        return None
    return {
        "pose_cam_from_model": objects.poses_cam_from_model[0].detach().cpu().numpy(),
        "score": float(objects.pose_scores[0].detach().cpu().item()),
    }
