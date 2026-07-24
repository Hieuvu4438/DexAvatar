"""Provider-neutral interfaces for later external datasets and observations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from phase2_refiner.data.cache_schema import CacheClip


@runtime_checkable
class ObservationProvider(Protocol):
    """Enrich a cache clip with 3D/camera/appearance observations and masks."""

    name: str

    def enrich(self, clip: CacheClip) -> CacheClip: ...


@runtime_checkable
class TargetProvider(Protocol):
    """Attach independently supervised sequence targets and provenance."""

    name: str

    def attach_targets(self, clip: CacheClip) -> CacheClip: ...


def apply_providers(
    clip: CacheClip,
    observations: tuple[ObservationProvider, ...] = (),
    target: TargetProvider | None = None,
) -> CacheClip:
    """Apply optional providers while enforcing the common schema after each step."""
    for provider in observations:
        clip = provider.enrich(clip)
        clip.validate()
    if target is not None:
        clip = target.attach_targets(clip)
        clip.validate()
    return clip
