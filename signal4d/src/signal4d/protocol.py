from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import load_yaml


class ProtocolViolation(RuntimeError):
    """Raised when a requested action would leak or invalidate the protocol."""


@dataclass(frozen=True)
class ProtocolGuard:
    track: str
    allow_test_fit: bool = True
    allow_test_calibration: bool = False
    require_complete_predictions: bool = True
    coordinate_convention: str = "opencv_x_right_y_down_z_forward"
    length_unit: str = "meter"

    @classmethod
    def from_config(cls, path: str | Path) -> ProtocolGuard:
        cfg = load_yaml(path)
        return cls(
            track=str(cfg["track"]),
            allow_test_fit=bool(cfg.get("allow_test_fit", True)),
            allow_test_calibration=bool(cfg.get("allow_test_calibration", False)),
            require_complete_predictions=bool(cfg.get("require_complete_predictions", True)),
            coordinate_convention=str(cfg.get("coordinate_convention", cls.coordinate_convention)),
            length_unit=str(cfg.get("length_unit", cls.length_unit)),
        )

    def validate_action(self, action: str, splits: str | Iterable[str]) -> None:
        split_set = {splits} if isinstance(splits, str) else set(splits)
        if action in {"calibrate", "tune", "select_hparam"} and "test" in split_set:
            raise ProtocolViolation(f"{action} is forbidden on the test split")
        if action == "fit" and "test" in split_set and not self.allow_test_fit:
            raise ProtocolViolation("fit is disabled on the test split by this protocol")

    def validate_metadata(self, coordinate_convention: str, length_unit: str) -> None:
        if coordinate_convention != self.coordinate_convention:
            raise ProtocolViolation(
                f"coordinate mismatch: {coordinate_convention} != {self.coordinate_convention}"
            )
        if length_unit != self.length_unit:
            raise ProtocolViolation(f"unit mismatch: {length_unit} != {self.length_unit}")
