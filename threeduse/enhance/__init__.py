"""Paper-facing Stage-2 components: ATC supervision and U-BAF appearance."""

from .appearance_compiler import CompiledAppearance, Stage2AppearanceCompiler
from .consensus_appearance import ConsensusStage2Appearance
from .effective_transition import (
    EffectiveTransition,
    extract_effective_transition,
    global_transition_condition,
    local_transition_condition,
)
from .scene_appearance_field import SceneAppearanceField
from .multiview_consensus import MultiViewOperatorConsensus
from .transition_calibrator import DeterministicTransitionCalibrator

# Paper-facing aliases keep the implementation names readable without changing
# any serialized module or buffer paths in released checkpoints.
AppearanceTransitionConsensus = MultiViewOperatorConsensus
UnderwaterBilateralAppearanceField = SceneAppearanceField

__all__ = [
    "AppearanceTransitionConsensus",
    "CompiledAppearance",
    "ConsensusStage2Appearance",
    "DeterministicTransitionCalibrator",
    "EffectiveTransition",
    "extract_effective_transition",
    "global_transition_condition",
    "local_transition_condition",
    "MultiViewOperatorConsensus",
    "SceneAppearanceField",
    "Stage2AppearanceCompiler",
    "UnderwaterBilateralAppearanceField",
]
