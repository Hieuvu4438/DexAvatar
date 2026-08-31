from __future__ import annotations

import os
import platform
import random

import numpy as np
import torch


def set_deterministic(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if deterministic:
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=True)


def runtime_metadata(device: torch.device | str) -> dict[str, object]:
    device = torch.device(device)
    metadata: dict[str, object] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": True,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(device)
        metadata["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return metadata
