"""PyTorch datasets for fixed-N quantum trajectories.

Each input trajectory contains the normalized collective spin values
``s_z(t_0), ..., s_z(t_{T-1})``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SingleNTrajectoryDataset(Dataset):
    """Return one full, fixed-N trajectory as a sequence-to-sequence sample.

    Parameters
    ----------
    sz:
        Array with shape ``(n_trajectories, n_time_points)``.  For the
        supplied N=8 data, this is ``(1000, 210)``.
    time_end:
        End of the *normalized* time feature.  Keep the default ``1.0`` when
        all trajectories use the same physical time grid, as they do here.

    Each item is ``(features, delta_sz)`` with shapes:

    - ``features``: ``(n_time_points - 1, 2)`` containing
      ``[current_sz, normalized_time]``;
    - ``delta_sz``: ``(n_time_points - 1,)`` containing
      ``sz[t + 1] - sz[t]``.
    """

    def __init__(self, sz: np.ndarray | torch.Tensor, time_end: float = 1.0):
        sz = torch.as_tensor(sz, dtype=torch.float32)

        if sz.ndim != 2:
            raise ValueError(
                "sz must have shape (n_trajectories, n_time_points); "
                f"received shape {tuple(sz.shape)}."
            )
        if sz.shape[0] == 0 or sz.shape[1] < 2:
            raise ValueError("sz needs at least one trajectory with two time points.")
        if not torch.isfinite(sz).all():
            raise ValueError("sz contains NaN or infinite values.")

        self.sz = sz.contiguous()
        self.time = torch.linspace(
            0.0,
            time_end,
            steps=self.sz.shape[1] - 1,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.sz.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory = self.sz[index]

        current_sz = trajectory[:-1]
        delta_sz = trajectory[1:] - current_sz

        features = torch.stack((current_sz, self.time), dim=-1)
        return features, delta_sz


class SingleNNextStateDataset(Dataset):
    """Return full trajectories with the bounded next state as the target.

    This dataset is intended for categorical/grid models.  Unlike
    :class:`SingleNTrajectoryDataset`, the target is ``sz[t + 1]`` itself,
    not an increment.  Predicting the next state on a grid bounded by
    ``[state_min, state_max]`` guarantees that autoregressive samples remain
    in the physical state space.

    ``features`` has shape ``(n_time_points - 1, 2)`` and contains
    ``[sz[t], normalized_time[t]]``.  ``next_sz`` has shape
    ``(n_time_points - 1,)``.
    """

    def __init__(
        self,
        sz: np.ndarray | torch.Tensor,
        time_end: float = 1.0,
        state_min: float = -1.0,
        state_max: float = 1.0,
    ) -> None:
        sz = torch.as_tensor(sz, dtype=torch.float32)

        if sz.ndim != 2:
            raise ValueError(
                "sz must have shape (n_trajectories, n_time_points); "
                f"received shape {tuple(sz.shape)}."
            )
        if sz.shape[0] == 0 or sz.shape[1] < 2:
            raise ValueError("sz needs at least one trajectory with two time points.")
        if not torch.isfinite(sz).all():
            raise ValueError("sz contains NaN or infinite values.")
        if state_min >= state_max:
            raise ValueError("state_min must be smaller than state_max.")
        if time_end <= 0:
            raise ValueError("time_end must be positive.")

        observed_min = sz.min().item()
        observed_max = sz.max().item()
        tolerance = 10 * torch.finfo(sz.dtype).eps
        if observed_min < state_min - tolerance or observed_max > state_max + tolerance:
            raise ValueError(
                "sz lies outside the requested state interval "
                f"[{state_min}, {state_max}]: observed "
                f"[{observed_min}, {observed_max}]."
            )

        self.sz = sz.contiguous()
        self.state_min = float(state_min)
        self.state_max = float(state_max)

        # The input at transition i is sz(t_i), so omit the final time rather
        # than stretching T - 1 source times over the full [0, time_end].
        self.time = torch.linspace(
            0.0,
            time_end,
            steps=self.sz.shape[1],
            dtype=torch.float32,
        )[:-1]

    def __len__(self) -> int:
        return self.sz.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory = self.sz[index]
        features = torch.stack((trajectory[:-1], self.time), dim=-1)
        return features, trajectory[1:]
