from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FamilyDelta:
    name: str
    delta: float
    sigma: float
    available: bool = True

    @property
    def wins(self) -> bool:
        return (
            self.available and math.isfinite(self.delta) and math.isfinite(self.sigma)
            and self.sigma >= 0 and self.delta < -2.0 * self.sigma
        )

    @property
    def loses(self) -> bool:
        return (
            self.available and math.isfinite(self.delta) and math.isfinite(self.sigma)
            and self.sigma >= 0 and self.delta > self.sigma
        )


def accept_ubody(
    families: list[FamilyDelta],
    *,
    changes_depth: bool,
    geometry_ok: bool,
    trust_ok: bool,
    minimum_active_families: int = 2,
) -> tuple[bool, list[str], list[str], str]:
    active = [family for family in families if family.available]
    if any(not math.isfinite(family.delta) or not math.isfinite(family.sigma) for family in active):
        return False, [], [], "GATE_NONFINITE_EVIDENCE"
    if any(family.sigma < 0 for family in active):
        return False, [], [], "GATE_INVALID_SIGMA"
    winners = [family.name for family in active if family.wins]
    losers = [family.name for family in active if family.loses]
    if len(active) < minimum_active_families:
        return False, winners, losers, "GATE_NOT_ENOUGH_ACTIVE_FAMILIES"
    if not geometry_ok:
        return False, winners, losers, "OFF_TARGET_DRIFT"
    if not trust_ok:
        return False, winners, losers, "TRUST_BOUND_HIT"
    if losers:
        return False, winners, losers, "GATE_FAMILY_REGRESSION"
    if len(winners) < 2:
        return False, winners, losers, "GATE_NOT_ENOUGH_WINS"
    if changes_depth and not ({"nlf3d", "dense_rgb"} & set(winners)):
        return False, winners, losers, "GATE_NO_3D_WIN"
    return True, winners, losers, "ACCEPTED"
