from __future__ import annotations

import re


GLOBAL = re.compile(r"\[(?P<method>[^\]]+)\]:\s+(?P<metric>[^:]+):\s+(?P<value>[0-9.+-eE]+)\s+\(mm\)")


def parse_global_metrics(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for match in GLOBAL.finditer(text):
        key = " ".join(match.group("metric").lower().split())
        result[key] = float(match.group("value"))
    required = {"tr left hand", "tr right hand", "tr above pelvis upper body"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Missing metrics: {sorted(missing)}")
    return result

