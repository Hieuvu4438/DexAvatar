"""Staged, fail-closed training primitives and checkpoint contracts."""

from .batch import (
    SupervisedWindowBatch,
    SupervisedWindowMetadata,
    load_supervised_windows,
    save_supervised_windows,
)
from .checkpoint import CheckpointMetadata, load_model_checkpoint, save_model_checkpoint
from .steps import ContactObjective, DiffusionObjective, contact_objective, diffusion_objective

__all__ = [
    "CheckpointMetadata",
    "ContactObjective",
    "DiffusionObjective",
    "SupervisedWindowBatch",
    "SupervisedWindowMetadata",
    "contact_objective",
    "diffusion_objective",
    "load_model_checkpoint",
    "load_supervised_windows",
    "save_model_checkpoint",
    "save_supervised_windows",
]
