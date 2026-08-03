"""GRU mixture-density network for one fixed-N trajectory ensemble."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TrajectoryGRUMDN(nn.Module):
    """Predict a Gaussian-mixture distribution for the next ``delta_sz``.

    The input at each time step is ``[current_sz, normalized_time]``.  The
    output parameters describe ``p(delta_sz | input history)`` and have shape
    ``(batch, sequence_length, mixtures)``.

    ``logits`` are deliberately returned before softmax: this makes the MDN
    negative-log-likelihood numerically stable through ``log_softmax``.
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        mixtures: int = 5,
        dropout: float = 0.1,
        min_scale: float = 1e-3,
    ) -> None:
        super().__init__()

        if input_size != 2:
            raise ValueError("The fixed-N dataset has exactly two input features.")
        if hidden_size < 1 or num_layers < 1 or mixtures < 1:
            raise ValueError("hidden_size, num_layers, and mixtures must be positive.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mixtures = mixtures
        self.min_scale = min_scale

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
            nn.Linear(hidden_size, 3 * mixtures),
        )

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, means, scales, hidden)`` for every input step."""
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(
                "x must have shape (batch, sequence_length, "
                f"{self.input_size}); received {tuple(x.shape)}."
            )

        output, hidden = self.gru(x, hidden)
        parameters = self.head(output)
        logits, means, raw_scales = torch.chunk(parameters, chunks=3, dim=-1)
        scales = F.softplus(raw_scales) + self.min_scale

        return logits, means, scales, hidden


class TrajectoryGRUGrid(nn.Module):
    """Predict a categorical distribution over bounded next-state values.

    The grid includes both endpoints, so every sampled next state lies in
    ``[state_min, state_max]``.  The model predicts the next state directly
    rather than an increment; a bounded increment alone would not prevent
    accumulated trajectories from leaving the physical interval.
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        grid_size: int = 401,
        dropout: float = 0.1,
        state_min: float = -1.0,
        state_max: float = 1.0,
    ) -> None:
        super().__init__()

        if input_size != 2:
            raise ValueError("The fixed-N dataset has exactly two input features.")
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive.")
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        if state_min >= state_max:
            raise ValueError("state_min must be smaller than state_max.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.grid_size = grid_size
        self.state_min = float(state_min)
        self.state_max = float(state_max)

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
        """Sample grid values from logits whose last dimension is the grid."""
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
    """Cross-entropy against the two grid points bracketing each target.

    Linear target weights avoid hard nearest-bin jumps and preserve the
    target value as the soft label's expectation.  The logits must have shape
    ``target.shape + (grid_size,)``.
    """
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
