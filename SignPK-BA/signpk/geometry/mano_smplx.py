from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .robustifiers import charbonnier, masked_mean


def root_align(points: Tensor, root_index: int = 0) -> Tensor:
    return points - points[..., root_index : root_index + 1, :]


@dataclass(frozen=True)
class HandCorrespondence:
    smplx_indices: Tensor
    mano_indices: Tensor

    def validate(self, smplx_vertex_count: int = 10475, mano_vertex_count: int = 778) -> None:
        if self.smplx_indices.ndim != 1 or self.mano_indices.ndim != 1:
            raise ValueError("correspondence indices must be vectors")
        if self.smplx_indices.shape != self.mano_indices.shape:
            raise ValueError("SMPL-X/MANO correspondence lengths differ")
        if self.smplx_indices.min() < 0 or self.smplx_indices.max() >= smplx_vertex_count:
            raise ValueError("invalid SMPL-X correspondence index")
        if self.mano_indices.min() < 0 or self.mano_indices.max() >= mano_vertex_count:
            raise ValueError("invalid MANO correspondence index")


def root_aligned_hand_vertex_loss(
    smplx_vertices: Tensor,
    mano_vertices: Tensor,
    correspondence: HandCorrespondence,
    valid: Tensor | None = None,
) -> Tensor:
    correspondence.validate(smplx_vertices.shape[-2], mano_vertices.shape[-2])
    smplx = smplx_vertices[..., correspondence.smplx_indices, :]
    mano = mano_vertices[..., correspondence.mano_indices, :]
    smplx = smplx - smplx.mean(-2, keepdim=True)
    mano = mano - mano.mean(-2, keepdim=True)
    return masked_mean(charbonnier(smplx - mano), valid)
