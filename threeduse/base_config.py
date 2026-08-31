import copy

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from threeduse.model import ThreeDUSEModelConfig

_stage1_base = MethodSpecification(
    config=TrainerConfig(
        method_name="3duse-stage1",
        save_only_latest_checkpoint=True,
        steps_per_eval_image=0,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=1000,
        max_num_iterations=15001,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                _target=FullImageDatamanager[DepthDataset],
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=ThreeDUSEModelConfig(
                num_steps=15001,
                ssim_lambda=0.2,
                lpips_lambda=0.00,
                medium_sh_degree=3,
                inject_noise_to_position=True,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5,
                    max_steps=15001,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025,
                    max_steps=15001,
                ),
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025 / 20,
                    max_steps=15001,
                ),
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.05,
                    max_steps=15001,
                ),
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.005,
                    max_steps=15001,
                ),
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.001,
                    max_steps=15001,
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=15001
                ),
            },
            "medium_feature_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025,
                    max_steps=15001,
                ),
            },
            "medium_feature_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00025 / 20,
                    max_steps=15001,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="3D-USE depth-prior reconstruction template.",
)


def _make_medium_rbf_method(
    method_name: str, max_steps: int = 15001
) -> MethodSpecification:
    method = MethodSpecification(
        config=copy.deepcopy(_stage1_base.config),
        description=(
            "3D-USE medium-aware reconstruction with MediumRBF and the "
            "underwater Gaussian compositor."
        ),
    )
    method.config.method_name = method_name
    method.config.max_num_iterations = max_steps
    method.config.steps_per_save = 15000
    method.config.steps_per_eval_all_images = 0 if max_steps > 15001 else 1000
    method.config.pipeline.model.num_steps = max_steps
    method.config.pipeline.model.medium_representation = "medium_rbf"
    method.config.pipeline.model.use_depth_gradient_rasterizer = False
    method.config.pipeline.model.use_depth_prior = True
    method.config.pipeline.model.depth_prior_lambda = 0.1
    method.config.pipeline.model.depth_prior_stop_step = 15000
    method.config.pipeline.model.medium_rbf_initial_width = 0.85
    method.config.pipeline.model.medium_rbf_topk = 4
    for optimizer_config in method.config.optimizers.values():
        scheduler = optimizer_config.get("scheduler", None)
        if scheduler is not None and hasattr(scheduler, "max_steps"):
            scheduler.max_steps = max_steps
    return method


stage1_template = _make_medium_rbf_method("3duse-stage1", max_steps=15001)

__all__ = ["stage1_template"]
