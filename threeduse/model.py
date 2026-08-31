# ruff: noqa: E741
# Copyright 2024 Huapeng Li, Wenxuan Song, Tianao Xu, Alexandre Elsig and Jonas KulhanekS. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for 3D-USE from WaterSplatting and Plenodium sources.

"""The released two-stage 3D-USE model.

Stage 1 reconstructs Gaussian object appearance together with MediumRBF
coefficients. Stage 2 freezes that reconstruction and optimizes ATC-supervised
U-BAF appearance for persistent enhanced novel-view rendering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB
from pytorch_msssim import MS_SSIM, SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from threeduse._torch_impl import quat_to_rotmat
from threeduse.sh import num_sh_bases, sparse_spherical_harmonics, spherical_harmonics
from threeduse.enhance.consensus_appearance import ConsensusStage2Appearance
from threeduse.enhance.effective_transition import (
    EffectiveTransition,
    GLOBAL_OPERATOR_DIM,
    LOCAL_OPERATOR_DIM,
    center_local_operator,
    extract_effective_transition,
    global_transition_condition,
    local_transition_condition,
)
from threeduse.enhance.multiview_consensus import MultiViewOperatorConsensus
from threeduse.enhance.scene_appearance_field import SceneAppearanceField
from threeduse.enhance.transition_calibrator import (
    DeterministicTransitionCalibrator,
)
from threeduse.enhance.uie_proposer import (
    load_frozen_uie_proposer,
    run_frozen_uie_proposer,
)
from threeduse.rendering.stage_renderers import (
    GradientPolicy,
    ReconstructionRender,
    StageRenderBundle,
    UnderwaterEnhancementRenderer,
    UnderwaterReconstructionRenderer,
)
from threeduse.rendering.underwater import rasterize_clear_from_underwater_meta

class SimplePSNR(nn.Module):
    """Small PSNR module to avoid importing torchmetrics during training/rendering."""

    def __init__(self, data_range: float = 1.0) -> None:
        super().__init__()
        self.data_range = float(data_range)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((pred.float() - target.float()) ** 2).clamp_min(1e-12)
        data_range = torch.tensor(self.data_range, device=mse.device, dtype=mse.dtype)
        return 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse)


def _make_lpips_metric(device: torch.device) -> Optional[nn.Module]:
    """Lazily create LPIPS for ns-eval without adding checkpoint state at init."""
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except Exception as exc:
        CONSOLE.print(f"[yellow]LPIPS metric unavailable: {exc}[/yellow]")
        return None
    return LearnedPerceptualImagePatchSimilarity(normalize=True).to(device).eval()


def quat_scale_to_covar_preci(
    quats: torch.Tensor,
    scales: torch.Tensor,
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Local replacement for the small gsplat utility used by noise injection."""

    covars = None
    precis = None
    rot = quat_to_rotmat(quats)
    if compute_covar:
        transform = rot * scales[..., None, :]
        covars = transform @ transform.transpose(-1, -2)
    if compute_preci:
        inv_scales = scales.clamp_min(1e-12).reciprocal()
        transform = rot * inv_scales[..., None, :]
        precis = transform @ transform.transpose(-1, -2)
    if triu:
        triu_idx = torch.triu_indices(3, 3, device=quats.device)
        if covars is not None:
            covars = covars[..., triu_idx[0], triu_idx[1]]
        if precis is not None:
            precis = precis[..., triu_idx[0], triu_idx[1]]
    return covars, precis


def interpolation_medium_rbf(
    T,
    medium_dc,
    medium_rest,
    centers: torch.Tensor,
    log_sigma: torch.Tensor,
    topk: int,
):
    """Isotropic MediumRBF interpolation for observer-conditioned medium SH."""

    medium_feature = torch.cat((medium_dc[:, None], medium_rest), dim=1)
    medium_size = medium_feature.shape
    anchors = medium_feature.reshape(*medium_size[:3], -1)
    pos = T.reshape(3).to(dtype=T.dtype).clamp(-1.5, 1.5)
    centers = centers.to(device=T.device, dtype=T.dtype)
    sigma = log_sigma.to(device=T.device, dtype=T.dtype).exp().clamp_min(1e-4)
    offsets = centers - pos[None, :]
    scores = -offsets.square().sum(dim=-1) / (2.0 * sigma.square())
    if 0 < topk < scores.shape[0]:
        keep = scores.topk(topk).indices
        masked_scores = scores.new_full(scores.shape, -torch.inf)
        scores = masked_scores.scatter(0, keep, scores[keep])
    weights = torch.softmax(scores, dim=0)
    anchor_mean = anchors.mean(dim=-1)
    local = torch.einsum("abck,k->abc", anchors - anchor_mean[..., None], weights)
    return anchor_mean + local


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float().flatten()
    y = y.float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().mean().sqrt() * y.square().mean().sqrt()
    return (x * y).mean() / denom.clamp_min(1e-6)


@dataclass
class ThreeDUSEModelConfig(ModelConfig):
    """Configuration shared by the two released 3D-USE stages."""

    _target: Type = field(default_factory=lambda: ThreeDUSEModel)
    num_steps: int = 15001
    """Number of steps to train the model"""
    warmup_length: int = 600
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.5
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_alpha_thresh_post: float = 0.1
    """threshold of opacity for post culling gaussians"""
    reset_alpha_thresh: float = 0.45
    """threshold of opacity for resetting alpha"""
    cull_scale_thresh: float = 10.
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 5
    """Every this many refinement steps, reset the alpha"""
    abs_grad_densification: bool = True
    """If True, use absolute gradient for densification"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.001
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    medium_sh_degree_interval: int = 1000
    """every n intervals turn on another medium sh degree"""
    clip_thresh: float = 0.01
    """minimum depth threshold"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 0
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    lpips_lambda: float = 0.00
    """weight of lpips loss"""
    stop_split_at: int = 10000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """

    medium_sh_degree: int = 3
    """degree of the spherical harmonics to use for the medium field"""
    medium_representation: tyro.conf.Suppress[
        Literal["medium_rbf", "nexus_kernel"]
    ] = "medium_rbf"
    """Medium estimator used before the underwater rasterizer.

    ``medium_rbf`` is the released representation. ``nexus_kernel`` is an
    equivalent legacy spelling retained only for existing checkpoint configs.
    """
    medium_rbf_initial_width: float = 0.85
    """Initial support width of the MediumRBF anchors."""
    medium_rbf_topk: int = 4
    """Number of MediumRBF anchors selected for each camera position."""
    medium_nexus_sigma: tyro.conf.Suppress[Optional[float]] = None
    """Compatibility alias for checkpoints created before the public rename."""
    medium_nexus_topk: tyro.conf.Suppress[Optional[int]] = None
    """Compatibility alias for checkpoints created before the public rename."""
    inject_noise_to_position: bool = False
    """If enabled, inject noise to the gaussians position"""
    noise_lr: float = 2e5
    """The noise learning rate"""
    use_depth_gradient_rasterizer: tyro.conf.Suppress[bool] = False
    """If enabled, use the rasterizer variant that backpropagates rendered depth."""
    stage1_renderer: tyro.conf.Suppress[Literal["gsplat_underwater"]] = (
        "gsplat_underwater"
    )
    """Final underwater gsplat renderer; retained in saved configs."""
    stage1_gsplat_projection_far_plane: float = 1.0e4
    """Projection/culling far plane for the underwater gsplat backend."""
    use_depth_prior: bool = False
    """Enable coarse Pearson alignment with the DA2 disparity prior."""
    depth_prior_lambda: float = 0.1
    """Weight of the coarse depth-prior loss."""
    depth_prior_stop_step: int = 15000
    """Iteration after which depth-prior supervision is disabled."""
    enable_stage2_enhancement: bool = False
    """Enable multi-view operator consensus and independent enhanced appearance."""
    stage2_transition_calibrator_checkpoint: str = ""
    """Paired-data transition calibrator conditioned on UIE proposals."""
    stage2_uie_proposer_checkpoint: str = ""
    """Frozen UIE proposer used only to construct captured-view operators."""
    stage2_bilateral_grid_res: int = 16
    """Resolution of the shared Gaussian/medium 4D appearance field."""
    stage2_bilateral_rank: int = 8
    """CP rank of the persistent appearance field."""
    stage2_feature_hidden_dim: int = 64
    """Decoder width of the persistent appearance field."""
    stage2_bilateral_tv_lambda: float = 1e-4
    """Weight of smoothness regularization on the U-BAF factors."""
    stage2_transport_log_ratio_bound: float = 1.3862943611198906
    """Symmetric bound on the two scalar enhanced-transport log ratios."""
    stage2_coordinate_quantile: float = 0.01
    """Robust tail fraction for the fixed 3D field domain."""
    stage2_loss_max_side: int = 384
    """Maximum side used by transition extraction and the UIE proposer."""
    stage2_transition_grid_long_side: int = 16
    """Long-side patch count of the local effective operator."""
    stage2_transition_global_ridge: float = 1e-3
    stage2_transition_local_ridge: float = 1e-2


class ThreeDUSEModel(Model):
    """Underwater Gaussian scene model shared by the two accepted stages."""

    config: ThreeDUSEModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def _medium_rbf_width(self) -> float:
        legacy = self.config.medium_nexus_sigma
        return float(
            self.config.medium_rbf_initial_width if legacy is None else legacy
        )

    def _medium_rbf_topk(self) -> int:
        legacy = self.config.medium_nexus_topk
        return int(self.config.medium_rbf_topk if legacy is None else legacy)

    def _depth_prior_settings(self) -> tuple[bool, float, int]:
        return (
            bool(self.config.use_depth_prior),
            float(self.config.depth_prior_lambda),
            int(self.config.depth_prior_stop_step),
        )

    def populate_modules(self):
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()

        self.medium_feature_dc = nn.Parameter(
            torch.tensor([[0., 0., 0.], [-5., -5., -5.], [-5., -5., -5.]]).reshape(3,3,1,1,1).repeat(1,1,2,2,2)
        )
        self.medium_feature_rest = nn.Parameter(
            torch.zeros(3, num_sh_bases(self.config.medium_sh_degree) - 1, 3).reshape(3,-1,3,1,1,1).repeat(1,1,1,2,2,2)
        )
        # ``medium_nexus_*`` are the historical state-dict keys used by the
        # released checkpoints. They implement the isotropic MediumRBF anchors
        # and must remain stable even though the public method name changed.
        coords_1d = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        cx, cy, cz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
        centers = torch.stack([cx, cy, cz], dim=-1).reshape(-1, 3)
        self.medium_nexus_centers = nn.Parameter(centers)
        self.medium_nexus_log_sigma = nn.Parameter(
            torch.full(
                (centers.shape[0],),
                math.log(self._medium_rbf_width()),
                dtype=torch.float32,
            )
        )
        self.register_buffer(
            "medium_representation_code",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = k_nearest_sklearn(means.data, 3)
        self.avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(self.avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )
        # metrics
        self.psnr = SimplePSNR(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.ssim_fn = MS_SSIM(
            data_range=1.0,
            size_average=True,
            channel=3,
            weights=[0.6, 0.3, 0.1],
        )
        self.lpips = None
        self.step = 0
        self.crop_box: Optional[OrientedBox] = None
        self.background_color = torch.nn.Parameter(torch.tensor([0.,0.,0.]))
        # Training-only frozen guidance stays outside the checkpoint module
        # tree.  Stage-2 checkpoints contain only persistent scene appearance
        # and the accumulated multi-view consensus.
        object.__setattr__(self, "stage2_prior", None)
        object.__setattr__(self, "stage2_uie_proposer", None)
        object.__setattr__(self, "stage2_view_guidance_cache", {})
        object.__setattr__(self, "_stage2_last_projection", None)
        object.__setattr__(self, "_stage2_last_gaussian_indices", None)
        object.__setattr__(self, "_stage2_last_raster_size", None)
        object.__setattr__(self, "_stage2_inference_only", False)
        self.stage2_prior_loaded = False
        self.stage2_reconstruction_frozen = False
        self.stage2_appearance = None
        self.stage2_consensus = None
        if self.config.enable_stage2_enhancement:
            # A continuous two-stage run constructs U-BAF before Stage-1 starts.
            # Its random field initialization must not perturb the global RNG
            # consumed later by stochastic Gaussian splitting.  Otherwise the
            # nominally identical 0--15K reconstruction develops a different
            # topology merely because an inactive enhancement module exists.
            cpu_rng_state = torch.random.get_rng_state()
            try:
                self.stage2_appearance = ConsensusStage2Appearance(
                    num_points,
                    grid_resolution=int(self.config.stage2_bilateral_grid_res),
                    rank=int(self.config.stage2_bilateral_rank),
                    hidden_dim=int(self.config.stage2_feature_hidden_dim),
                    coordinate_quantile=float(self.config.stage2_coordinate_quantile),
                    cache_compiled=True,
                    max_transport_log_ratio=float(
                        self.config.stage2_transport_log_ratio_bound
                    ),
                )
                self.stage2_consensus = MultiViewOperatorConsensus(num_points)
            finally:
                torch.random.set_rng_state(cpu_rng_state)
        self.reconstruction_renderer = UnderwaterReconstructionRenderer()
        self.enhancement_renderer = UnderwaterEnhancementRenderer()
        if self.config.enable_stage2_enhancement:
            self._stage2_freeze_reconstruction_params()

    def train(self, mode: bool = True):  # type: ignore[override]
        # Periodic eval must never reuse appearance compiled before an optimizer
        # update.  The compiler itself permits cache reads only under no_grad.
        if mode and self.stage2_appearance is not None:
            self.stage2_appearance.clear_cache()
        return super().train(mode)

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    @property
    def stage2_appearance_field(self) -> Optional[nn.Module]:
        """Persistent field owned by EnhancementState."""

        if self.stage2_appearance is None:
            return None
        return self.stage2_appearance.field

    def _stage2_parameter_groups(self) -> Dict[str, List[Parameter]]:
        """Separate persistent colour/medium state from scalar transport."""

        if self.stage2_appearance is None:
            raise RuntimeError("Stage-2 appearance is not initialized")
        appearance: List[Parameter] = []
        transport: List[Parameter] = []
        for name, parameter in self.stage2_appearance.named_parameters():
            if name.startswith("compiler.transport_"):
                transport.append(parameter)
            else:
                appearance.append(parameter)
        if not appearance or len(transport) != 2:
            raise RuntimeError(
                "Stage-2 must expose appearance parameters and exactly two "
                "transport residuals"
            )
        return {
            "stage2_appearance": appearance,
            "stage2_transport": transport,
        }

    def _stage2_active(self) -> bool:
        return self.config.enable_stage2_enhancement

    def _stage2_freeze_reconstruction_params(self) -> None:
        if not self.config.enable_stage2_enhancement:
            return
        for parameter in self.gauss_params.parameters():
            parameter.requires_grad_(False)
        self.medium_feature_dc.requires_grad_(False)
        self.medium_feature_rest.requires_grad_(False)
        self.medium_nexus_centers.requires_grad_(False)
        self.medium_nexus_log_sigma.requires_grad_(False)
        self.background_color.requires_grad_(False)
        self.stage2_reconstruction_frozen = True

    def _assert_stage2_parameter_ownership(self) -> None:
        """Fail immediately if Stage-2 can mutate reconstruction state."""

        if not self._stage2_active():
            return
        leaking = [
            name
            for name, parameter in self.named_parameters()
            if not name.startswith("stage2_appearance.")
            and name != "device_indicator_param"
            and parameter.requires_grad
        ]
        if leaking:
            raise RuntimeError(
                "Stage-2 parameter ownership violation; reconstruction parameters remain "
                f"trainable: {leaking[:8]}"
            )

    def _stage2_load_prior(self) -> None:
        """Load the paired residual calibrator and frozen UIE proposer."""

        if self.stage2_prior_loaded:
            return
        calibrator_checkpoint = self.config.stage2_transition_calibrator_checkpoint
        if not calibrator_checkpoint:
            raise ValueError(
                "Stage-2 requires pipeline.model.stage2-transition-calibrator-checkpoint"
            )
        if not self.config.stage2_uie_proposer_checkpoint:
            raise ValueError(
                "Stage-2 requires "
                "pipeline.model.stage2-uie-proposer-checkpoint"
            )
        prior, metadata = DeterministicTransitionCalibrator.from_checkpoint(
            calibrator_checkpoint
        )
        if prior.target_mode != "mlp":
            raise ValueError(
                "The Stage-2 mainline accepts only the learned paired residual "
                "calibrator; support lookup/barycenter checkpoints are obsolete"
            )
        if prior.prediction_mode != "proposal_residual":
            raise ValueError(
                "The Stage-2 mainline requires prediction_mode='proposal_residual'"
            )
        prior.to(self.device).eval().requires_grad_(False)
        proposer = load_frozen_uie_proposer(
            self.config.stage2_uie_proposer_checkpoint,
            device=self.device,
        )
        object.__setattr__(self, "stage2_prior", prior)
        object.__setattr__(self, "stage2_uie_proposer", proposer)
        if self.stage2_consensus is None:
            raise RuntimeError("Stage-2 consensus state is not initialized")
        self.stage2_consensus.set_local_scale(prior.local_scale)
        self.stage2_prior_loaded = True
        CONSOLE.log(
            "Loaded the paired-data transition calibrator and frozen UIE "
            f"proposal ({metadata.get('num_train_pairs', 'unknown')} paired samples)"
        )

    @staticmethod
    def _stage2_cached_image_index(batch: dict, fallback: int) -> int:
        value = batch.get("image_idx", fallback)
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].item())
        return int(value)

    @torch.no_grad()
    def _stage2_prepare_consensus(self, datamanager, step: int) -> None:
        """Compile all captured training views into scene-global/local guidance."""

        self._stage2_load_prior()
        prior = self.stage2_prior
        proposer = self.stage2_uie_proposer
        consensus = self.stage2_consensus
        if not isinstance(prior, DeterministicTransitionCalibrator):
            raise RuntimeError("Invalid Stage-2 residual calibrator")
        if proposer is None or consensus is None:
            raise RuntimeError("Stage-2 guidance is not initialized")
        # A fresh continuous run constructs Stage-2 modules before Stage-1
        # densification.  Resize only at delayed activation, after topology is
        # locked, so the persistent local buffers match final Gaussian IDs.
        consensus.resize_num_gaussians(self.num_points)
        if (
            bool(consensus.global_ready.detach().cpu().item())
            and bool(self.stage2_view_guidance_cache)
        ):
            return
        if not hasattr(datamanager, "cached_train"):
            raise RuntimeError("Stage-2 requires cached original training images")

        records: list[tuple[int, EffectiveTransition, EffectiveTransition, Tensor]] = []
        global_targets: list[Tensor] = []
        for fallback, batch in enumerate(datamanager.cached_train):
            observed = batch["image"]
            if observed.dtype == torch.uint8:
                observed = observed.float() / 255.0
            observed = observed[..., :3].to(self.device)
            if observed.ndim == 4:
                if observed.shape[0] != 1:
                    raise ValueError("Stage-2 cached training batches must contain one view")
                observed = observed[0]
            captured = self._extract_stage2_transition(observed, observed)
            proposed_image = run_frozen_uie_proposer(
                proposer,
                observed,
                max_side=int(self.config.stage2_loss_max_side),
            )
            proposal = self._extract_stage2_transition(observed, proposed_image)
            global_target = prior.global_target(
                global_transition_condition(captured, proposal)
            )
            image_index = self._stage2_cached_image_index(batch, fallback)
            records.append((image_index, captured, proposal, global_target))
            global_targets.append(global_target[0])

        if not records:
            raise RuntimeError("Cannot build Stage-2 consensus from zero views")
        scene_global = (
            consensus.global_target.detach()
            if bool(consensus.global_ready.detach().cpu().item())
            else consensus.set_global(torch.stack(global_targets))
        )
        cache: dict[int, dict[str, Tensor]] = {}
        for image_index, captured, proposal, view_global in records:
            target_global = scene_global.to(view_global)
            local_condition = local_transition_condition(
                captured,
                proposal,
                target_global,
            )
            local_target = prior.local_target(local_condition)
            local_target, removed_center = center_local_operator(local_target)
            # Low-texture water is appearance evidence too.  Confidence here
            # means visibility/support only; it must not erase gain or bias.
            confidence = torch.ones_like(local_target[..., :1])
            cache[image_index] = {
                "local_target": local_target.detach().cpu(),
                "local_confidence": confidence.detach().cpu(),
                "proposal_global": proposal.global_operator.detach().cpu(),
                "proposal_local": proposal.local_operator.detach().cpu(),
                "view_global_target": view_global.detach().cpu(),
                "removed_local_center": removed_center.detach().cpu(),
            }
        object.__setattr__(self, "stage2_view_guidance_cache", cache)
        if int(consensus.local_views.max().detach().cpu().item()) == 0:
            was_training = self.training
            self.eval()
            try:
                cameras = datamanager.train_dataset.cameras
                if len(cameras) != len(records):
                    raise RuntimeError(
                        "Cached Stage-2 views and training cameras do not align"
                    )
                for dataset_position, (image_index, _, _, _) in enumerate(records):
                    self.get_outputs_for_camera(
                        cameras[dataset_position : dataset_position + 1]
                    )
                    projection = self._stage2_last_projection
                    gaussian_indices = self._stage2_last_gaussian_indices
                    raster_size = self._stage2_last_raster_size
                    if (
                        projection is None
                        or gaussian_indices is None
                        or raster_size is None
                    ):
                        raise RuntimeError(
                            "Consensus precomputation missed a raster context"
                        )
                    width, height, tile_size = raster_size
                    view = cache[image_index]
                    consensus.backproject_and_update(
                        view["local_target"].to(self.device),
                        view["local_confidence"].to(self.device),
                        projection,
                        width=int(width),
                        height=int(height),
                        tile_size=int(tile_size),
                        gaussian_indices=gaussian_indices,
                    )
            finally:
                self.train(was_training)
        covered = (consensus.local_views >= 2).float().mean()
        CONSOLE.log(
            "Compiled one effect-space scene consensus and "
            f"{len(records)} captured-view local operator maps at step {step} "
            f"(global dispersion={float(consensus.global_dispersion):.4f}, "
            f"multi-view Gaussian coverage={float(covered):.3f})"
        )

    def load_state_dict(  # type: ignore[override]
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        """Restore Stage-1 or the single accepted consensus Stage-2 schema."""

        self.step = self.config.num_steps
        state_dict = dict(state_dict)
        obsolete_prefixes = (
            "stage2_clean_appearance.",
            "stage2_lut_appearance.",
            "stage2_palette_appearance.",
            "stage2_adapter.",
        )
        obsolete = [
            key for key in state_dict if key.startswith(obsolete_prefixes)
        ]
        if obsolete and self.config.enable_stage2_enhancement:
            raise RuntimeError(
                "This checkpoint belongs to a removed Stage-2 implementation. "
                "Restart from the frozen Stage-1 checkpoint; obsolete keys include "
                + ", ".join(obsolete[:4])
            )

        representation_key = "medium_representation_code"
        if representation_key not in state_dict:
            state_dict[representation_key] = (
                self.medium_representation_code.detach().clone()
            )
        else:
            loaded_code = int(
                state_dict[representation_key].detach().cpu().item()
            )
            configured_code = int(
                self.medium_representation_code.detach().cpu().item()
            )
            if loaded_code != configured_code:
                raise RuntimeError(
                    "Checkpoint/config medium mismatch: checkpoint code "
                    f"{loaded_code}, configured code {configured_code}"
                )

        if "means" in state_dict:
            for name in (
                "means",
                "scales",
                "quats",
                "features_dc",
                "features_rest",
                "opacities",
            ):
                state_dict[f"gauss_params.{name}"] = state_dict.pop(name)
        if "gauss_params.means" not in state_dict:
            raise RuntimeError("Checkpoint is missing gauss_params.means")
        newp = int(state_dict["gauss_params.means"].shape[0])
        for parameter in self.gauss_params.values():
            new_shape = (newp,) + tuple(parameter.shape[1:])
            if tuple(parameter.shape) != new_shape:
                parameter.data = torch.zeros(
                    new_shape,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
        if self.stage2_consensus is not None:
            self.stage2_consensus.resize_num_gaussians(newp)
            consensus_prefix = "stage2_consensus."
            loaded_local = state_dict.get(consensus_prefix + "local_mean")
            if loaded_local is not None and int(loaded_local.shape[0]) != newp:
                # Before Stage-2 activation the checkpoint contains only the
                # seed-sized, all-zero placeholder buffers while Stage-1 may
                # already have densified hundreds of thousands of Gaussians.
                # Those placeholders carry no learned state and must be
                # regenerated at the restored topology.  A ready consensus,
                # in contrast, is tied to Gaussian IDs and a mismatch is a
                # genuinely invalid checkpoint.
                ready = state_dict.get(consensus_prefix + "global_ready")
                is_ready = (
                    bool(ready.detach().cpu().item())
                    if ready is not None
                    else False
                )
                if is_ready:
                    raise RuntimeError(
                        "Ready Stage-2 consensus/Gaussian topology mismatch: "
                        f"consensus={int(loaded_local.shape[0])}, gaussians={newp}"
                    )
                for name in ("local_mean", "local_weight", "local_views"):
                    state_dict[consensus_prefix + name] = getattr(
                        self.stage2_consensus, name
                    ).detach().clone()

        has_stage2_state = any(
            key.startswith(("stage2_appearance.", "stage2_consensus."))
            for key in state_dict
        )
        has_stage2_bounds = (
            "stage2_appearance.compiler.field.scene_bounds_ready" in state_dict
        )
        incompatible = super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
        if strict:
            allowed_missing: set[str] = set()
            stage2_prefixes = ("stage2_appearance.", "stage2_consensus.")
            if self.config.enable_stage2_enhancement and not has_stage2_state:
                # Stage 2 intentionally starts from a complete Stage-1 model.
                allowed_missing.update(
                    key
                    for key in incompatible.missing_keys
                    if key.startswith(stage2_prefixes)
                )
            elif self.config.enable_stage2_enhancement and not has_stage2_bounds:
                allowed_missing.update(
                    {
                        "stage2_appearance.compiler.field.scene_position_min",
                        "stage2_appearance.compiler.field.scene_position_max",
                        "stage2_appearance.compiler.field.scene_bounds_ready",
                    }
                )
            missing = sorted(
                key
                for key in incompatible.missing_keys
                if key not in allowed_missing
            )
            unexpected = sorted(incompatible.unexpected_keys)
            if missing or unexpected:
                raise RuntimeError(
                    "3D-USE checkpoint state mismatch: "
                    f"missing={missing[:8]}, unexpected={unexpected[:8]}"
                )

        if self.config.enable_stage2_enhancement:
            field = self.stage2_appearance_field
            if isinstance(field, SceneAppearanceField) and not has_stage2_bounds:
                sampling_positions = [
                    self.means.detach(),
                    self.medium_nexus_centers.detach(),
                ]
                field.set_scene_bounds(
                    torch.cat(sampling_positions, dim=0),
                    quantile=float(self.config.stage2_coordinate_quantile),
                )
            if self.stage2_appearance is not None:
                self.stage2_appearance.clear_cache()
            self._stage2_freeze_reconstruction_params()
        return incompatible

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        if self.config.enable_stage2_enhancement:
            return
        if step < 100:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            projected = self._stage1_gsplat_means2d
            if self.config.abs_grad_densification:
                assert hasattr(projected, "absgrad")
                self.xys_grad_abs = projected.absgrad[0]
                projected_grads = self.xys_grad_abs.detach()
            else:
                assert projected.grad is not None
                projected_grads = projected.grad[0].detach()
            # gsplat reports derivatives in pixel coordinates. Convert each
            # axis independently to normalized screen space before applying
            # the densification threshold.
            screen_scale = projected_grads.new_tensor(
                [0.5 * self.last_size[1], 0.5 * self.last_size[0]]
            )
            grads = (projected_grads * screen_scale).norm(dim=-1)
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.depths_accum = self.depths
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]
                self.depths_accum[visible_mask] = self.depths[visible_mask] + self.depths_accum[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.config.enable_stage2_enhancement:
            return
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Split or cull only after every training image has been observed
            # following the most recent opacity reset.
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = self.step < self.config.stop_split_at and (
                self.step % reset_interval
                > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                # then we densify
                assert (
                    self.xys_grad_norm is not None
                    and self.vis_counts is not None
                    and self.max_2Dsize is not None
                )
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()

                splits = (
                    self.scales.exp().max(dim=-1).values
                    > self.config.densify_size_thresh
                ).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (
                        self.max_2Dsize > self.config.split_screen_size
                    ).squeeze()
                splits &= high_grads

                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (
                    self.scales.exp().max(dim=-1).values
                    <= self.config.densify_size_thresh
                ).squeeze()
                dups &= high_grads

                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat(
                            [param.detach(), split_params[name], dup_params[name]],
                            dim=0,
                        )
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # Prune the original Gaussian after creating its split copies.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )
                deleted_mask = self.cull_gaussians(splits_mask)
            elif (
                self.step >= self.config.stop_split_at
                and self.config.continue_cull_post_densification
            ):
                deleted_mask = self.cull_gaussians()
            else:
                # Stop pruning after refinement when post-culling is disabled.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

                # reset the exp of optimizer
                for key in [
                    "medium_feature_dc",
                    "medium_feature_rest",
                ]:
                    optim = optimizers.optimizers[key]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    if "exp_avg" in param_state:
                        param_state["exp_avg"] = torch.zeros_like(
                            param_state["exp_avg"]
                        )
                        param_state["exp_avg_sq"] = torch.zeros_like(
                            param_state["exp_avg_sq"]
                        )

            if (
                self.step < self.config.stop_split_at
                and self.step % reset_interval == self.config.refine_every
            ):
                # Reset value is set to be reset_alpha_thresh
                reset_value = self.config.reset_alpha_thresh
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(
                        torch.tensor(reset_value, device=self.device)
                    ).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

                self.inject_noise_to_position(optimizers, step)

            self.xys_grad_norm = None
            self.vis_counts = None
            self.depths_accum = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        # cull transparent ones
        if self.step < self.config.stop_split_at:
            cull_alpha_thresh = self.config.cull_alpha_thresh
        else:
            cull_alpha_thresh = self.config.cull_alpha_thresh_post
        culls = (torch.sigmoid(self.opacities) < cull_alpha_thresh).squeeze()
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    @torch.no_grad()
    def inject_noise_to_position(self,optimers: Optimizers,step):
        if (not self.config.inject_noise_to_position) or step < self.config.warmup_length:
            return
        lr = optimers.optimizers["means"].param_groups[0]["lr"]
        assert step == self.step
        opacities = torch.sigmoid(self.opacities.flatten())
        scales = torch.exp(self.scales)
        covars, _ = quat_scale_to_covar_preci(
            self.quats,
            scales,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        noise = (
            torch.randn_like(self.means)
            * (op_sigmoid(1 - opacities)).unsqueeze(-1)
            * lr
            * self.config.noise_lr
        )
        noise = torch.einsum("bij,bj->bi", covars, noise)
        self.means.add_(noise)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        if self.config.enable_stage2_enhancement:
            if training_callback_attributes.pipeline is None:
                raise RuntimeError("Stage-2 scene consensus requires the training pipeline")
            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    self._stage2_prepare_consensus,
                    args=[training_callback_attributes.pipeline.datamanager],
                )
            )
        # ATC stores local targets by Gaussian index, so Stage-2 topology is
        # fixed and only Stage 1 runs split/cull callbacks.
        allow_refinement = not self.config.enable_stage2_enhancement
        if allow_refinement:
            # The order of these matters.
            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    self.after_train,
                )
            )
            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    self.refinement_after,
                    update_every_num_iters=self.config.refine_every,
                    args=[
                        training_callback_attributes.optimizers,
                    ],
                )
            )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        if self.config.enable_stage2_enhancement:
            self._stage2_freeze_reconstruction_params()
            self._assert_stage2_parameter_ownership()
            return self._stage2_parameter_groups()
        gps = self.get_gaussian_param_groups()
        gps["medium_feature_dc"] = [self.medium_feature_dc]
        gps["medium_feature_rest"] = [
            self.medium_feature_rest,
            self.medium_nexus_centers,
            self.medium_nexus_log_sigma,
        ]
        if self.config.enable_stage2_enhancement:
            gps.update(self._stage2_parameter_groups())
        return gps

    def _get_downscale_factor(self):
        if self.training:
            downscale = 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
            return downscale
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        return image

    def get_outputs(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Render one camera with the Stage-1 underwater reconstruction path."""
        if not isinstance(camera, Cameras):
            raise TypeError(
                f"camera must be a Cameras instance, got {type(camera).__name__}"
            )
        assert camera.shape[0] == 1, "Only one camera at a time"
        if self.config.enable_stage2_enhancement:
            self._stage2_freeze_reconstruction_params()

        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)

        R = camera.camera_to_worlds[0, :3, :3]
        T = camera.camera_to_worlds[0, :3, 3:4]
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv

        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        self.last_fx = camera.fx.item()
        self.last_fy = camera.fy.item()

        # Keep the directional-medium convention compatible with Plenodium.
        # This direction only indexes the medium SH field; Gaussian projection
        # continues to use ``viewmat`` and the selected rasterizer below.
        medium_y = torch.linspace(
            0.0, float(H), H, device=self.device, dtype=R.dtype
        )
        medium_x = torch.linspace(
            0.0, float(W), W, device=self.device, dtype=R.dtype
        )
        yy, xx = torch.meshgrid(medium_y, medium_x, indexing="ij")
        yy = (yy - cy) / camera.fy.item()
        xx = (xx - cx) / camera.fx.item()
        directions = torch.stack([yy, xx, -torch.ones_like(xx)], dim=-1)
        directions = directions / torch.linalg.norm(
            directions, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        directions = directions @ R

        using_modern_stage1 = self.config.stage1_renderer == "gsplat_underwater"
        depth_prior_enabled, _, depth_prior_stop_step = self._depth_prior_settings()
        depth_prior_needs_grad = (
            self.training
            and torch.is_grad_enabled()
            and depth_prior_enabled
            and self.step < depth_prior_stop_step
        )
        if using_modern_stage1:
            ray_y = (
                torch.arange(H, device=self.device, dtype=R.dtype) + 0.5 - cy
            ) / camera.fy.item()
            ray_x = (
                torch.arange(W, device=self.device, dtype=R.dtype) + 0.5 - cx
            ) / camera.fx.item()
            ray_yy, ray_xx = torch.meshgrid(ray_y, ray_x, indexing="ij")
            stage1_ray_depth_scales = torch.sqrt(
                ray_xx.square() + ray_yy.square() + 1.0
            )[..., None]
            # ABI placeholder only.  The active CUDA compositor always uses
            # the original Plenodium infinite water-column tail.
            stage1_far_depths = -torch.ones(
                (H, W, 1), device=self.device, dtype=R.dtype
            )
        else:
            stage1_ray_depth_scales = None
            stage1_far_depths = None

        directions_flat = directions.view(-1, 3)
        outputs_shape = directions.shape[:-1]
        n_medium = min(self.step // self.config.medium_sh_degree_interval, self.config.medium_sh_degree)
        if self.config.medium_representation not in {"medium_rbf", "nexus_kernel"}:
            raise ValueError(f"Unknown medium representation: {self.config.medium_representation}")
        self.medium_feature = interpolation_medium_rbf(
            T,
            self.medium_feature_dc,
            self.medium_feature_rest,
            self.medium_nexus_centers,
            self.medium_nexus_log_sigma,
            self._medium_rbf_topk(),
        )
        medium_base_out = sparse_spherical_harmonics(
            n_medium, directions_flat, self.medium_feature
        )
        medium_rgb, medium_bs, medium_attn = medium_base_out.chunk(3, 0)
        medium_rgb = torch.clamp(
            medium_rgb.view(*outputs_shape, -1) + 0.5, min=0.0
        )
        medium_bs = self.sigma_activation(medium_bs.view(*outputs_shape, -1))
        medium_attn = self.sigma_activation(
            medium_attn.view(*outputs_shape, -1)
        )

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                if using_modern_stage1:
                    assert stage1_far_depths is not None
                    # Legacy infinite-tail semantics: an empty ray contains
                    # the complete learned medium color and no object depth.
                    rgb = medium_rgb
                    depth = torch.zeros_like(stage1_far_depths)
                else:
                    rgb = medium_rgb
                    depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
                empty_object = torch.zeros_like(rgb)
                outputs = {
                    "rgb": rgb,
                    "stage1_rgb": rgb,
                    "stage1_object": empty_object,
                    "stage1_medium": rgb,
                    "stage1_clear_linear": empty_object,
                    "depth": depth,
                    "accumulation": accumulation,
                    "background": medium_rgb,
                    "rgb_object": empty_object,
                    "rgb_clear": empty_object,
                    "rgb_clear_unclamp": empty_object,
                    "rgb_clear_clamp": empty_object,
                    "rgb_medium": rgb,
                    "pred_image": rgb,
                    "medium_rgb": medium_rgb,
                    "medium_bs": medium_bs,
                    "medium_attn": medium_attn,
                    "medium_feat": self.medium_feature,
                }
                if using_modern_stage1:
                    outputs["object_depth_z"] = depth
                    outputs["object_path"] = depth
                    outputs["depth_valid"] = torch.zeros_like(depth)
                    outputs["water_exit_path"] = torch.zeros_like(stage1_far_depths)
                if self._stage2_active():
                    if self.stage2_appearance is None:
                        raise RuntimeError("Stage-2 appearance is not initialized")
                    enhanced_medium, _ = self.stage2_appearance.medium_appearance(
                        medium_rgb,
                        T.reshape(3),
                        strength=1.0,
                    )
                    outputs.update(
                        {
                            "stage2_source_rgb": rgb,
                            "enhanced_object": empty_object,
                            "enhanced_object_rgb": empty_object,
                            "enhanced_medium_rgb": enhanced_medium,
                            "enhanced_medium_source": enhanced_medium,
                            "enhanced_underwater_rgb": enhanced_medium,
                            "enhanced_clear_linear": empty_object,
                            "enhanced_clear_display": empty_object,
                            "enhanced_clear_clamp": empty_object,
                            "enhanced_rgb": enhanced_medium,
                            "enhanced_accumulation": accumulation,
                            "stage2_expected_depth": depth,
                        }
                    )
                    outputs["pred_image"] = enhanced_medium
                camera.rescale_output_resolution(camera_downscale)
                return outputs
        else:
            crop_ids = None

        if crop_ids is not None and crop_ids.sum() != 0:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        block_width = 16

        if not using_modern_stage1:
            self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
                means_crop,
                torch.exp(scales_crop),
                1,
                quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                viewmat.squeeze()[:3, :],
                camera.fx.item(),
                camera.fy.item(),
                cx,
                cy,
                H,
                W,
                block_width,
                clip_thresh=self.config.clip_thresh,
            )
            self.depths = depths.detach()
        camera.rescale_output_resolution(camera_downscale)

        if not using_modern_stage1 and (self.radii).sum() == 0:
            rgb = medium_rgb
            depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
            empty_object = torch.zeros_like(rgb)
            outputs = {
                "rgb": rgb,
                "stage1_rgb": rgb,
                "stage1_object": empty_object,
                "stage1_medium": rgb,
                "stage1_clear_linear": empty_object,
                "depth": depth,
                "accumulation": accumulation,
                "background": medium_rgb,
                "rgb_object": empty_object,
                "rgb_clear": empty_object,
                "rgb_clear_unclamp": empty_object,
                "rgb_clear_clamp": empty_object,
                "rgb_medium": medium_rgb,
                "pred_image": rgb,
                "medium_rgb": medium_rgb,
                "medium_bs": medium_bs,
                "medium_attn": medium_attn,
                "medium_feat": self.medium_feature,
            }
            return outputs

        if not using_modern_stage1 and self.training and self.xys.requires_grad:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n_sh = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n_sh, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)
            base_colors_crop = SH2RGB(features_dc_crop)
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])
            base_colors_crop = rgbs

        if not using_modern_stage1:
            assert (num_tiles_hit > 0).any()
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop)
            if using_modern_stage1:
                opacities = opacities.squeeze(-1)
            else:
                opacities = opacities * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
            if using_modern_stage1:
                opacities = opacities.squeeze(-1)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if using_modern_stage1:
            assert stage1_far_depths is not None
            assert stage1_ray_depth_scales is not None
            K = torch.tensor(
                [
                    [self.last_fx, 0.0, cx],
                    [0.0, self.last_fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                device=self.device,
                dtype=means_crop.dtype,
            )
            if bool(self._stage2_inference_only):
                if self.training or torch.is_grad_enabled():
                    raise RuntimeError("Stage-2-only rendering is an inference-only API")
                if not self._stage2_active() or self.stage2_appearance is None:
                    raise RuntimeError("Stage-2-only rendering requires an active enhancement checkpoint")
                context = self.reconstruction_renderer.prepare_context(
                    means=means_crop,
                    scales=torch.exp(scales_crop),
                    quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                    opacities=opacities,
                    viewmat=viewmat,
                    intrinsics=K,
                    medium_rgb=medium_rgb,
                    beta_bs=medium_bs,
                    beta_attn=medium_attn,
                    width=W,
                    height=H,
                    far_depths=stage1_far_depths,
                    ray_depth_scales=stage1_ray_depth_scales,
                    tile_size=block_width,
                    near_plane=self.config.clip_thresh,
                    projection_far_plane=self.config.stage1_gsplat_projection_far_plane,
                    antialiased=self.config.rasterize_mode == "antialiased",
                )
                compiled = self.stage2_appearance.object_appearance(
                    rgbs.detach(),
                    base_colors_crop.detach(),
                    means_crop.detach(),
                    strength=1.0,
                )
                enhanced_medium, _ = self.stage2_appearance.medium_appearance(
                    medium_rgb, T.reshape(3), strength=1.0
                )
                beta_bs_plus, beta_attn_plus, _ = self.stage2_appearance.transport(
                    context.beta_bs, context.beta_attn, strength=1.0
                )
                enhanced = self.enhancement_renderer(
                    compiled.enhanced_linear, context, medium_source=enhanced_medium,
                    beta_bs_source=beta_bs_plus, beta_attn_source=beta_attn_plus,
                    policy=GradientPolicy.stage2(),
                )
                enhanced_clear_display = enhanced.clear_linear / (
                    enhanced.clear_linear + 1.0
                )
                return {
                    "pred_image": enhanced.composite,
                    "enhanced_rgb": enhanced.composite,
                    "enhanced_underwater_rgb": enhanced.composite,
                    "enhanced_object": enhanced.object,
                    "enhanced_medium_rgb": enhanced.medium,
                    "enhanced_clear_linear": enhanced.clear_linear,
                    "enhanced_clear_display": enhanced_clear_display,
                    "enhanced_accumulation": enhanced.alpha,
                    "stage2_expected_depth": enhanced.depth,
                }
            stage1_render = self.reconstruction_renderer(
                means=means_crop,
                scales=torch.exp(scales_crop),
                quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                opacities=opacities,
                colors=rgbs,
                viewmat=viewmat,
                intrinsics=K,
                medium_rgb=medium_rgb,
                beta_bs=medium_bs,
                beta_attn=medium_attn,
                width=W,
                height=H,
                far_depths=stage1_far_depths,
                ray_depth_scales=stage1_ray_depth_scales,
                tile_size=block_width,
                near_plane=self.config.clip_thresh,
                projection_far_plane=self.config.stage1_gsplat_projection_far_plane,
                antialiased=self.config.rasterize_mode == "antialiased",
                absgrad=self.config.abs_grad_densification,
                optical_depth_grad_scale=(
                    1.0
                    if self.config.use_depth_gradient_rasterizer
                    or depth_prior_needs_grad
                    else 0.0
                ),
            )
            rgb = stage1_render.composite
            rgb_object = stage1_render.object
            rgb_clear = stage1_render.clear_linear
            rgb_medium = stage1_render.medium
            alpha = stage1_render.alpha
            depth_im = stage1_render.depth
            gsplat_meta = stage1_render.context.projection

            self._stage1_gsplat_means2d = gsplat_meta["means2d"]
            if self.training and self._stage1_gsplat_means2d.requires_grad:
                self._stage1_gsplat_means2d.retain_grad()
            self.xys = self._stage1_gsplat_means2d[0]
            radii = gsplat_meta["radii"]
            if (
                radii.ndim == self._stage1_gsplat_means2d.ndim
                and radii.shape[-1] == 2
            ):
                radii_xy = radii[0]
                self.radii = torch.where(
                    (radii_xy > 0).all(dim=-1),
                    radii_xy.amax(dim=-1),
                    torch.zeros_like(radii_xy[..., 0]),
                )
            else:
                self.radii = radii[0]
            self.depths = gsplat_meta["depths"][0].detach()
            self.xys_grad_abs = torch.zeros_like(self.xys)
        else:
            self.xys_grad_abs = torch.zeros_like(self.xys)
            rasterize_fn = (
                rasterize_gaussians2
                if self.config.use_depth_gradient_rasterizer or depth_prior_needs_grad
                else rasterize_gaussians
            )

            rgb_object, rgb_clear, rgb_medium, depth_im, alpha = rasterize_fn(  # type: ignore
                self.xys,
                self.xys_grad_abs,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                rgbs,
                opacities,
                medium_rgb,
                medium_bs,
                medium_attn,
                H,
                W,
                block_width,
                background=medium_rgb,
                return_alpha=True,
                step=self.step,
            )
            rgb = rgb_object + rgb_medium
            depth_im = depth_im[..., None]
            alpha = alpha[..., None]

        depth_valid = alpha > 1e-6
        stage2_aux: dict[str, torch.Tensor] = {}
        if using_modern_stage1:
            assert stage1_far_depths is not None
            assert stage1_ray_depth_scales is not None
            enhancement_render = None
            if self._stage2_active():
                self._assert_stage2_parameter_ownership()
                if self.stage2_appearance is None:
                    raise RuntimeError("Stage-2 appearance is not initialized")

                strength = 1.0
                # Reconstruction appearance is private to the observed-image
                # head.  The operator objective must not rewrite Stage-1 SH or
                # shared geometry through its source colour, so C+ is learned
                # from detached reconstruction appearance.
                source_rgbs = rgbs.detach()
                compiled = self.stage2_appearance.object_appearance(
                    source_rgbs,
                    base_colors_crop.detach(),
                    means_crop.detach(),
                    strength=strength,
                )
                enhanced_medium, medium_aux = (
                    self.stage2_appearance.medium_appearance(
                        medium_rgb,
                        T.reshape(3),
                        strength=strength,
                    )
                )
                beta_bs_plus, beta_attn_plus, transport_aux = (
                    self.stage2_appearance.transport(
                        stage1_render.context.beta_bs,
                        stage1_render.context.beta_attn,
                        strength=strength,
                    )
                )
                # The appearance/operator objective never owns reconstruction.
                enhancement_render = self.enhancement_renderer(
                    compiled.enhanced_linear,
                    stage1_render.context,
                    medium_source=enhanced_medium,
                    beta_bs_source=beta_bs_plus,
                    beta_attn_source=beta_attn_plus,
                    policy=GradientPolicy.stage2(),
                )
                stage2_aux = dict(compiled.auxiliary)
                stage2_aux.update(medium_aux)
                stage2_aux.update(transport_aux)

                # The frozen Stage-1 projection is the exact visibility operator
                # used to lift one view's local guidance into Gaussian space.
                object.__setattr__(
                    self,
                    "_stage2_last_projection",
                    dict(stage1_render.context.projection),
                )
                gaussian_indices = (
                    torch.nonzero(crop_ids, as_tuple=False).flatten()
                    if crop_ids is not None
                    else torch.arange(
                        means_crop.shape[0],
                        device=means_crop.device,
                        dtype=torch.long,
                    )
                )
                object.__setattr__(
                    self,
                    "_stage2_last_gaussian_indices",
                    gaussian_indices.detach(),
                )
                object.__setattr__(
                    self,
                    "_stage2_last_raster_size",
                    (
                        int(stage1_render.context.width),
                        int(stage1_render.context.height),
                        int(stage1_render.context.tile_size),
                    ),
                )

                if not all(
                    torch.isfinite(tensor).all()
                    for tensor in (
                        enhancement_render.composite,
                        enhancement_render.object,
                        enhancement_render.medium,
                        enhancement_render.alpha,
                        enhancement_render.depth,
                    )
                ):
                    raise FloatingPointError("Non-finite Stage-2 underwater render")

            bundle = StageRenderBundle(
                reconstruction=stage1_render,
                enhancement=enhancement_render,
            )
            outputs = bundle.as_output_dict(
                enhanced_prediction=enhancement_render is not None,
            )
            outputs.update(
                {
                    "background": medium_rgb,
                    "medium_rgb": medium_rgb,
                    "medium_bs": medium_bs,
                    "medium_attn": medium_attn,
                    "medium_feat": self.medium_feature,
                    "object_depth_z": stage1_render.depth,
                    "object_path": stage1_render.depth * stage1_ray_depth_scales,
                    "depth_valid": depth_valid.to(stage1_render.depth.dtype),
                    "water_exit_path": torch.zeros_like(stage1_far_depths),
                }
            )
        else:
            # The legacy renderer is retained for Stage-1 checkpoint
            # compatibility only.  Stage-2 construction rejects this backend.
            depth_im = torch.where(
                depth_valid,
                depth_im / alpha.clamp_min(1e-6),
                depth_im.detach().max(),
            )
            clear_linear = rgb_clear
            outputs = {
                "rgb": rgb,
                "pred_image": rgb,
                "stage1_rgb": rgb,
                "stage1_object": rgb_object,
                "stage1_medium": rgb_medium,
                "stage1_clear_linear": clear_linear,
                "depth": depth_im,
                "accumulation": alpha,
                "background": medium_rgb,
                "rgb_object": rgb_object,
                "rgb_medium": rgb_medium,
                "rgb_clear": clear_linear / (clear_linear + 1.0),
                "rgb_clear_unclamp": clear_linear,
                "rgb_clear_clamp": clear_linear.clamp(0.0, 1.0),
                "medium_rgb": medium_rgb,
                "medium_bs": medium_bs,
                "medium_attn": medium_attn,
                "medium_feat": self.medium_feature,
            }

        if stage2_aux:
            if "enhanced_rgb" in outputs:
                rendered_delta = outputs["enhanced_rgb"] - outputs["stage1_rgb"].detach()
                outputs["stage2_rgb_delta_abs_mean"] = rendered_delta.abs().mean()
                outputs["stage2_rgb_delta_mean"] = rendered_delta.mean(dim=(0, 1))
                outputs["stage2_rgb_delta_std"] = rendered_delta.std(
                    dim=(0, 1), unbiased=False
                )
            for name, value in stage2_aux.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    outputs[f"stage2_{name}"] = value
        return outputs

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., 3:4]
            return alpha * image[..., :3] + (1.0 - alpha) * background
        return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self._stage2_active() and "enhanced_rgb" in outputs:
            metrics_dict["stage2_enhanced_mean"] = outputs["enhanced_rgb"].mean()
            if "stage2_rgb_delta_abs_mean" in outputs:
                metrics_dict["stage2_rgb_delta_abs_mean"] = outputs["stage2_rgb_delta_abs_mean"]
            for name in (
                "stage2_transition_global_target_error",
                "stage2_transition_local_target_error",
                "stage2_transition_coverage",
                "stage2_source_descriptor_mismatch",
                "stage2_transition_global_magnitude",
                "stage2_transition_local_magnitude",
                "stage2_target_global_magnitude",
                "stage2_consensus_visible_gaussians",
                "stage2_consensus_global_dispersion",
            ):
                if name in outputs:
                    metrics_dict[name] = outputs[name]

        metrics_dict["gaussian_count"] = self.num_points
        for i in range(3):
            metrics_dict[f"medium_attn_{i}"] = outputs["medium_attn"][:, :, i].mean()
            metrics_dict[f"medium_bs_{i}"] = outputs["medium_bs"][:, :, i].mean()
            metrics_dict[f"medium_rgb_{i}"] = outputs["medium_rgb"][:, :, i].mean()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Compute the objective of the selected 3D-USE stage."""
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        if not self._stage2_active():
            return self._stage1_loss_dict(outputs, batch, gt_img, outputs["stage1_rgb"])
        image_index = batch.get("image_idx")
        if torch.is_tensor(image_index):
            image_index = int(image_index.reshape(-1)[0].item())
        elif image_index is not None:
            image_index = int(image_index)
        return self._stage2_loss_dict(outputs, gt_img, image_index)

    def _stage1_loss_dict(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        gt_img: torch.Tensor,
        pred_img: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute Stage-1 reconstruction losses against the raw underwater image."""

        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        weight = (pred_img.detach() + 1e-3).reciprocal()
        recon_loss = torch.abs(weight * (pred_img - gt_img)).mean()

        self.ssim_fn = self.ssim_fn.to(gt_img.device)
        simloss = 1 - self.ssim_fn(
            (weight * gt_img).permute(2, 0, 1)[None, ...],
            (weight * pred_img).permute(2, 0, 1)[None, ...],
        )

        lpips_loss = (
            self.lpips(
                gt_img.permute(2, 0, 1)[None, ...],
                pred_img.permute(2, 0, 1)[None, ...].clamp(0.0, 1.0),
            )
            if self.config.lpips_lambda > 0 and self.lpips is not None
            else torch.tensor(0.0).to(self.device)
        )
        main_loss = (
            (1 - self.config.ssim_lambda - self.config.lpips_lambda) * recon_loss
            + self.config.ssim_lambda * simloss
            + self.config.lpips_lambda * lpips_loss
        )
        depth_prior_enabled, depth_prior_weight, depth_prior_stop_step = (
            self._depth_prior_settings()
        )
        depth_prior_loss = (
            depth_prior_weight * self._depth_prior_loss(outputs["depth"], batch)
            if depth_prior_enabled and self.step < depth_prior_stop_step
            else torch.tensor(0.0).to(self.device)
        )

        return {
            "main_loss": main_loss,
            "depth_prior_loss": depth_prior_loss,
        }

    def _depth_prior_loss(
        self,
        rendered_depth: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        """Coarse Pearson loss between DA2 disparity and rendered disparity."""

        # Nerfstudio's ``DepthDataset`` exposes metric/pseudo depth under
        # ``depth_image``. Keep ``depth`` as a fallback for older cached
        # batches.
        depth_key = "depth_image" if "depth_image" in batch else "depth" if "depth" in batch else None
        if depth_key is None:
            return rendered_depth.sum() * 0.0
        pseudo_depth = self._downscale_if_required(batch[depth_key]).to(self.device).float()
        if pseudo_depth.ndim == 3 and pseudo_depth.shape[-1] == 1:
            pseudo_depth = pseudo_depth[..., 0]
        pred = rendered_depth
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred[..., 0]
        if pseudo_depth.shape != pred.shape:
            pseudo_depth = F.interpolate(
                pseudo_depth[None, None, ...],
                size=pred.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        valid = torch.isfinite(pseudo_depth) & torch.isfinite(pred) & (pseudo_depth > 0) & (pred > 0)
        if int(valid.sum().item()) < 32:
            return pred.sum() * 0.0
        pseudo_valid = pseudo_depth[valid]
        pred_disp = 1.0 / (pred[valid].clamp_min(1e-6) * 10.0 + 1.0)
        # Pearson correlation is affine-scale invariant, so no pseudo-depth
        # normalization is needed.
        return 1.0 - pearson_corr(pseudo_valid, pred_disp)


    def _extract_stage2_transition(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ):
        """Use one operator contract for paired guidance and rendered students."""

        return extract_effective_transition(
            source,
            target,
            local_grid_long_side=int(self.config.stage2_transition_grid_long_side),
            max_side=int(self.config.stage2_loss_max_side),
            global_ridge=float(self.config.stage2_transition_global_ridge),
            local_ridge=float(self.config.stage2_transition_local_ridge),
        )

    def _stage2_loss_dict(
        self,
        outputs: Dict[str, torch.Tensor],
        observed_img: torch.Tensor,
        view_index: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Distill calibrated effects without an enhanced RGB pseudo-label."""

        required = {"stage1_rgb", "enhanced_rgb"}
        missing = required.difference(outputs)
        if missing:
            raise RuntimeError(
                "Consensus Stage-2 renderer missed: " + ", ".join(sorted(missing))
            )
        self._stage2_load_prior()
        prior = self.stage2_prior
        consensus = self.stage2_consensus
        if not isinstance(prior, DeterministicTransitionCalibrator):
            raise RuntimeError("Invalid Stage-2 residual calibrator")
        if consensus is None or not bool(consensus.global_ready.item()):
            raise RuntimeError("Scene consensus was not prepared before Stage-2 loss")
        if view_index is None:
            raise RuntimeError("Stage-2 requires the captured training-view index")
        if int(view_index) not in self.stage2_view_guidance_cache:
            raise KeyError(f"No calibrated local guidance for training view {view_index}")

        projection = self._stage2_last_projection
        gaussian_indices = self._stage2_last_gaussian_indices
        raster_size = self._stage2_last_raster_size
        if (
            projection is None
            or gaussian_indices is None
            or raster_size is None
        ):
            raise RuntimeError("Stage-2 forward did not retain its frozen raster context")
        width, height, tile_size = raster_size
        target_map, support_map = consensus.rasterized_local_target(
            projection,
            width=int(width),
            height=int(height),
            tile_size=int(tile_size),
            gaussian_indices=gaussian_indices,
        )

        student = self._extract_stage2_transition(
            outputs["stage1_rgb"].detach(),
            outputs["enhanced_rgb"],
        )
        global_target = consensus.global_target.to(student.global_operator)
        global_error = (
            student.global_operator - global_target
        ) / prior.global_scale.to(student.global_operator).clamp_min(1.0e-4)
        global_loss = F.smooth_l1_loss(
            global_error,
            torch.zeros_like(global_error),
        )

        grid_size = tuple(student.local_operator.shape[1:3])
        local_target_grid = F.interpolate(
            target_map.permute(2, 0, 1)[None],
            size=grid_size,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        support_grid = F.interpolate(
            support_map.permute(2, 0, 1)[None],
            size=grid_size,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).clamp(0.0, 1.0)
        student_local, _ = center_local_operator(student.local_operator)
        local_error = (
            student_local - local_target_grid
        ) / prior.local_scale.to(student_local).clamp_min(1.0e-4)
        local_per_cell = F.smooth_l1_loss(
            local_error,
            torch.zeros_like(local_error),
            reduction="none",
        ).mean(dim=-1, keepdim=True)
        if float(support_grid.sum().detach()) > 0.0:
            local_loss = (
                local_per_cell * support_grid
            ).sum() / support_grid.sum().clamp_min(1.0e-6)
        else:
            local_loss = local_per_cell.sum() * 0.0

        losses = {
            "stage2_global_consensus": global_loss,
            "stage2_local_consensus": local_loss,
        }
        if (
            self.stage2_appearance is not None
            and float(self.config.stage2_bilateral_tv_lambda) > 0.0
        ):
            losses["stage2_field_tv"] = (
                float(self.config.stage2_bilateral_tv_lambda)
                * self.stage2_appearance.tv_loss()
            )

        outputs["stage2_transition_global_target_error"] = global_loss.detach()
        outputs["stage2_transition_local_target_error"] = local_loss.detach()
        outputs["stage2_transition_coverage"] = support_grid.mean().detach()
        outputs["stage2_transition_global_magnitude"] = (
            student.global_operator.norm(dim=-1).mean().detach()
        )
        outputs["stage2_transition_local_magnitude"] = (
            student_local.norm(dim=-1).mean().detach()
        )
        outputs["stage2_target_global_magnitude"] = (
            global_target.norm(dim=-1).mean().detach()
        )
        outputs["stage2_consensus_visible_gaussians"] = (
            consensus.local_views[gaussian_indices].ge(2).sum().to(
                outputs["enhanced_rgb"]
            )
        )
        outputs["stage2_consensus_global_dispersion"] = (
            consensus.global_dispersion.detach()
        )
        return losses

    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device), obb_box=obb_box)
        return outs  # type: ignore

    @torch.no_grad()
    def get_stage2_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, torch.Tensor]:
        """Render only the persistent enhancement branch for deployment/FPS.

        Projection, tile intersection, and sorting run once; the Stage-1 pixel
        compositor is deliberately skipped. Training and ``get_outputs`` keep
        their existing dual-semantic contract.
        """
        previous = bool(self._stage2_inference_only)
        object.__setattr__(self, "_stage2_inference_only", True)
        try:
            return self.get_outputs_for_camera(camera, obb_box=obb_box)
        finally:
            object.__setattr__(self, "_stage2_inference_only", previous)

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Compute evaluation metrics and visualization images.

        Args:
            outputs: Outputs of the model.
            batch: Evaluation batch containing the reference image.

        Returns:
            The scalar metrics and visualization image dictionaries.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])

        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)

        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(predicted_rgb.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)

        output_gt_rgb = gt_rgb.cpu()

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        if self.lpips is None:
            self.lpips = _make_lpips_metric(gt_rgb.device)
        lpips = self.lpips(gt_rgb, predicted_rgb) if self.lpips is not None else torch.tensor(float("nan"))

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "gt": output_gt_rgb,
            "stage1_medium": outputs["stage1_medium"],
            "stage1_object": outputs["stage1_object"],
            "depth": outputs["depth"],
            "stage1_rgb": outputs["stage1_rgb"],
            "stage1_clear_display": outputs["rgb_clear"],
            "rgb_clear_clamp": outputs["rgb_clear_clamp"],
        }
        if "enhanced_object" in outputs:
            images_dict["enhanced_object"] = outputs["enhanced_object"]
            images_dict["enhanced_medium"] = outputs["enhanced_medium_rgb"]
            images_dict["enhanced_rgb"] = outputs["enhanced_rgb"]
        return metrics_dict, images_dict
