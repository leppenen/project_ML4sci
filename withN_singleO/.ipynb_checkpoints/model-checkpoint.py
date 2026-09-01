"""Particle-number-conditioned GRU with a bounded categorical state grid."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConditionedTrajectoryGRUGrid(nn.Module):
    """Predict ``p(s_z(t+1) | history, time, N)`` on a bounded grid.

    Every recurrent input is ``[current_sz, normalized_time, N/n_reference]``.
    Treating N as a continuous conditioning variable permits interpolation and
    numerical extrapolation, although accuracy beyond the training range must
    be tested empirically.
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        grid_size: int = 401,
        dropout: float = 0.1,
        state_min: float = -1.0,
        state_max: float = 1.0,
        n_reference: float = 18.0,
    ) -> None:
        super().__init__()

        if input_size != 3:
            raise ValueError(
                "The conditioned dataset has three features: state, time, and N."
            )
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive.")
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        if state_min >= state_max:
            raise ValueError("state_min must be smaller than state_max.")
        if n_reference <= 0:
            raise ValueError("n_reference must be positive.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.grid_size = grid_size
        self.state_min = float(state_min)
        self.state_max = float(state_max)
        self.n_reference = float(n_reference)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, grid_size),
        )
        self.register_buffer(
            "state_grid",
            torch.linspace(state_min, state_max, steps=grid_size),
        )

    def encode_particle_number(self, particle_number: torch.Tensor) -> torch.Tensor:
        """Apply the same N normalization used during training."""
        if torch.any(particle_number <= 0):
            raise ValueError("particle_number must be positive.")
        return particle_number / self.n_reference

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(grid_logits, hidden)`` for every input step."""
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(
                "x must have shape (batch, sequence_length, "
                f"{self.input_size}); received {tuple(x.shape)}."
            )

        output, hidden = self.gru(x, hidden)
        return self.head(output), hidden

    def sample_next_state(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample bounded state-grid values from categorical logits."""
        if logits.shape[-1] != self.grid_size:
            raise ValueError(
                f"Expected {self.grid_size} grid logits; "
                f"received {logits.shape[-1]}."
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        probabilities = torch.softmax(logits / temperature, dim=-1)
        original_shape = probabilities.shape[:-1]
        indices = torch.multinomial(
            probabilities.reshape(-1, self.grid_size),
            num_samples=1,
            generator=generator,
        )
        return self.state_grid[indices.squeeze(-1)].reshape(original_shape)


def interpolated_grid_nll(
    target: torch.Tensor,
    logits: torch.Tensor,
    state_min: float = -1.0,
    state_max: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy against the two grid points bracketing each target."""
    if logits.shape[:-1] != target.shape:
        raise ValueError(
            "logits must have shape target.shape + (grid_size,); "
            f"received target {tuple(target.shape)} and logits "
            f"{tuple(logits.shape)}."
        )
    if logits.shape[-1] < 2:
        raise ValueError("The state grid needs at least two points.")
    if state_min >= state_max:
        raise ValueError("state_min must be smaller than state_max.")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")

    tolerance = 10 * torch.finfo(target.dtype).eps
    if torch.any(target < state_min - tolerance) or torch.any(
        target > state_max + tolerance
    ):
        raise ValueError(
            f"target contains values outside [{state_min}, {state_max}]."
        )

    grid_size = logits.shape[-1]
    position = (
        (target.clamp(state_min, state_max) - state_min)
        * (grid_size - 1)
        / (state_max - state_min)
    )
    lower_index = position.floor().long()
    upper_index = (lower_index + 1).clamp_max(grid_size - 1)
    upper_weight = position - lower_index
    lower_weight = 1.0 - upper_weight

    log_probabilities = F.log_softmax(logits, dim=-1)
    lower_log_probability = log_probabilities.gather(
        dim=-1, index=lower_index.unsqueeze(-1)
    ).squeeze(-1)
    upper_log_probability = log_probabilities.gather(
        dim=-1, index=upper_index.unsqueeze(-1)
    ).squeeze(-1)

    loss = -(
        lower_weight * lower_log_probability
        + upper_weight * upper_log_probability
    )
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
