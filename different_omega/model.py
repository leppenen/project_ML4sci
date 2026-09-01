"""Bounded GRU-grid model conditioned on N and Omega/Omega_c."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConditionedNOmegaTrajectoryGRUGrid(nn.Module):
    """Predict p(s_z(t+1) | history, time, N, omega_ratio)."""

    def __init__(self, input_size: int = 4, hidden_size: int = 128,
                 num_layers: int = 2, grid_size: int = 401, dropout: float = 0.1,
                 state_min: float = -1.0, state_max: float = 1.0,
                 n_reference: float = 18.0, omega_reference: float = 1.0) -> None:
        super().__init__()
        if input_size != 4:
            raise ValueError("Inputs are [state, time, N, omega_ratio].")
        if hidden_size < 1 or num_layers < 1 or grid_size < 2:
            raise ValueError("hidden_size, num_layers, and grid_size must be positive.")
        if state_min >= state_max or n_reference <= 0 or omega_reference <= 0:
            raise ValueError("Invalid state interval or conditioning reference.")
        self.input_size, self.hidden_size, self.num_layers, self.grid_size = input_size, hidden_size, num_layers, grid_size
        self.state_min, self.state_max = float(state_min), float(state_max)
        self.n_reference, self.omega_reference = float(n_reference), float(omega_reference)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, grid_size))
        self.register_buffer("state_grid", torch.linspace(state_min, state_max, grid_size))

    def encode_particle_number(self, value: torch.Tensor) -> torch.Tensor:
        if torch.any(value <= 0):
            raise ValueError("particle_number must be positive.")
        return value / self.n_reference

    def encode_omega_ratio(self, value: torch.Tensor) -> torch.Tensor:
        if torch.any(value <= 0):
            raise ValueError("omega_ratio must be positive.")
        return value / self.omega_reference

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(f"x must have shape (batch, sequence, {self.input_size}).")
        output, hidden = self.gru(x, hidden)
        return self.head(output), hidden

    def sample_next_state(self, logits: torch.Tensor, temperature: float = 1.0,
                          generator: torch.Generator | None = None) -> torch.Tensor:
        if logits.shape[-1] != self.grid_size or temperature <= 0:
            raise ValueError("Invalid logits shape or temperature.")
        probabilities = torch.softmax(logits / temperature, dim=-1)
        shape = probabilities.shape[:-1]
        indices = torch.multinomial(probabilities.reshape(-1, self.grid_size), 1, generator=generator)
        return self.state_grid[indices.squeeze(-1)].reshape(shape)


def interpolated_grid_nll(target: torch.Tensor, logits: torch.Tensor,
                          state_min: float = -1.0, state_max: float = 1.0,
                          reduction: str = "mean") -> torch.Tensor:
    if logits.shape[:-1] != target.shape or logits.shape[-1] < 2:
        raise ValueError("logits must have shape target.shape + (grid_size,).")
    if state_min >= state_max or reduction not in {"none", "mean", "sum"}:
        raise ValueError("Invalid state interval or reduction.")
    position = (target.clamp(state_min, state_max) - state_min) * (logits.shape[-1] - 1) / (state_max - state_min)
    lower = position.floor().long()
    upper = (lower + 1).clamp_max(logits.shape[-1] - 1)
    upper_weight = position - lower
    log_prob = F.log_softmax(logits, dim=-1)
    loss = -(1 - upper_weight) * log_prob.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
    loss -= upper_weight * log_prob.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss
