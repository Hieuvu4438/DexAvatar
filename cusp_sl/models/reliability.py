"""Joint-token reliability model and held-out temperature calibration."""

from __future__ import annotations

import torch
from torch import nn


class ReliabilityCalibrator(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 96, temporal_layers: int = 3):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_size)
        self.joint_embedding = nn.Embedding(51, hidden_size)
        layers = []
        for _ in range(temporal_layers):
            layers.extend(
                (
                    nn.Conv1d(hidden_size, hidden_size, 3, padding=1, groups=1),
                    nn.GELU(),
                    nn.GroupNorm(8, hidden_size),
                )
            )
        self.temporal = nn.Sequential(*layers)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[2] != 51:
            raise ValueError("features must have shape [B,T,51,F]")
        value = self.input(features)
        joints = torch.arange(51, device=features.device)
        value = value + self.joint_embedding(joints)[None, None]
        b, t, j, h = value.shape
        temporal = value.permute(0, 2, 3, 1).reshape(b * j, h, t)
        temporal = self.temporal(temporal)
        value = temporal.reshape(b, j, h, t).permute(0, 3, 1, 2)
        return self.output(value).squeeze(-1)


class TemperatureScaler(nn.Module):
    """A positive scalar temperature fitted only on validation logits."""

    def __init__(self, initial: float = 1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(initial)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        logits = logits.detach().float().reshape(-1)
        labels = labels.detach().float().reshape(-1)
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach())

