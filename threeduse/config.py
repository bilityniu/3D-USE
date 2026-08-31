"""Public Nerfstudio entry points for the released two-stage 3D-USE method."""

from __future__ import annotations

import copy
import os
from pathlib import Path

from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from threeduse.base_config import stage1_template


def _environment_int(name: str, default: int) -> int:
    """Read a release-facing integer environment variable."""

    value = os.environ.get(name, str(default))
    return int(value)


def _environment_path(name: str) -> str:
    """Read a release-facing path environment variable."""

    return os.environ.get(name, "")


STAGE1_NUM_STEPS = _environment_int("THREEDUSE_STAGE1_NUM_STEPS", 15001)
STAGE2_END_STEP = _environment_int("THREEDUSE_STAGE2_END_STEP", 20001)
STAGE2_CALIBRATOR = _environment_path("THREEDUSE_STAGE2_CALIBRATOR")
STAGE2_UIE_PROPOSER_CHECKPOINT = _environment_path(
    "THREEDUSE_STAGE2_UIE_PROPOSER_CHECKPOINT"
)

if min(STAGE1_NUM_STEPS, STAGE2_END_STEP) < 2:
    raise ValueError("Stage iteration boundaries must be positive")
if STAGE2_END_STEP <= STAGE1_NUM_STEPS:
    raise ValueError("Stage-2 must end after the Stage-1 iteration boundary")


def _set_scheduler_horizon(specification: MethodSpecification, steps: int) -> None:
    for optimizer_config in specification.config.optimizers.values():
        scheduler = optimizer_config.get("scheduler")
        if scheduler is not None and hasattr(scheduler, "max_steps"):
            scheduler.max_steps = steps


# Stage 1 reconstructs the Gaussian scene and the observer-conditioned,
# direction-dependent MediumRBF coefficients used by the underwater compositor.
stage1 = MethodSpecification(
    config=copy.deepcopy(stage1_template.config),
    description=(
        "3D-USE Stage 1: medium-aware Gaussian reconstruction with MediumRBF "
        "and coarse monocular-depth supervision."
    ),
)
stage1.config.method_name = "3duse-stage1"
stage1.config.max_num_iterations = STAGE1_NUM_STEPS
stage1.config.steps_per_save = STAGE1_NUM_STEPS - 1
stage1.config.steps_per_eval_all_images = 0
stage1.config.pipeline.model.num_steps = STAGE1_NUM_STEPS
stage1.config.pipeline.model.medium_representation = "medium_rbf"
stage1.config.pipeline.model.use_depth_gradient_rasterizer = False
stage1.config.pipeline.model.use_depth_prior = True
stage1.config.pipeline.model.depth_prior_lambda = 0.1
stage1.config.pipeline.model.depth_prior_stop_step = min(
    15000, STAGE1_NUM_STEPS - 1
)
stage1.config.pipeline.model.stop_split_at = min(10000, STAGE1_NUM_STEPS - 1)
stage1.config.pipeline.model.continue_cull_post_densification = True
_set_scheduler_horizon(stage1, STAGE1_NUM_STEPS)


# Stage 2 freezes all Stage-1 reconstruction parameters. Appearance Transition
# Consensus (ATC) constructs fixed targets, and U-BAF stores their realization
# in persistent Gaussian and medium appearance.
stage2 = MethodSpecification(
    config=copy.deepcopy(stage1_template.config),
    description=(
        "3D-USE Stage 2: Appearance Transition Consensus (ATC) supervision "
        "realized by the Underwater Bilateral Appearance Field (U-BAF)."
    ),
)
stage2.config.method_name = "3duse-stage2"
stage2.config.max_num_iterations = STAGE2_END_STEP
stage2.config.steps_per_save = 500
stage2.config.steps_per_eval_image = 0
stage2.config.steps_per_eval_all_images = 0
stage2.config.log_gradients = False
stage2.config.pipeline.model.num_steps = STAGE2_END_STEP
stage2.config.pipeline.model.medium_representation = "medium_rbf"
stage2.config.pipeline.model.enable_stage2_enhancement = True
stage2.config.pipeline.model.stage2_transition_calibrator_checkpoint = (
    STAGE2_CALIBRATOR
)
stage2.config.pipeline.model.stage2_uie_proposer_checkpoint = (
    STAGE2_UIE_PROPOSER_CHECKPOINT
)
stage2.config.pipeline.model.stage2_bilateral_grid_res = 16
stage2.config.pipeline.model.stage2_bilateral_rank = 8
stage2.config.pipeline.model.stage2_feature_hidden_dim = 64
stage2.config.pipeline.model.stage2_transport_log_ratio_bound = 1.3862943611198906
stage2.config.pipeline.model.stage2_loss_max_side = 384
stage2.config.pipeline.model.num_downscales = 0
stage2.config.pipeline.model.stop_split_at = 0
stage2.config.pipeline.model.use_depth_prior = False
stage2.config.pipeline.datamanager.dataparser = ColmapDataParserConfig(
    images_path=Path("images_wb"),
    depths_path=Path("depth"),
    colmap_path=Path("colmap/sparse/0"),
    load_3D_points=False,
)
stage2.config.optimizers = {
    "stage2_appearance": {
        "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
        "scheduler": None,
    },
    "stage2_transport": {
        "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
        "scheduler": None,
    },
    "camera_opt": {
        "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=0.0, max_steps=STAGE2_END_STEP
        ),
    },
}
if hasattr(stage2.config, "load_optimizer"):
    stage2.config.load_optimizer = False
if hasattr(stage2.config, "load_scheduler"):
    stage2.config.load_scheduler = False


__all__ = ["stage1", "stage2"]
