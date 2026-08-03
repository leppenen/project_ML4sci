"""Datasets for one trajectory model conditioned on particle number N."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiNNextStateDataset(Dataset):
    """Return ``[current_sz, normalized_time, normalized_N]`` sequences.

    Parameters
    ----------
    sz:
        Trajectories with shape ``(n_trajectories, n_time_points)``.
    particle_numbers:
        One positive particle number for each trajectory.
    n_reference:
        Scale used to encode particle number as ``N / n_reference``.  This
        value must also be used at generation time and is saved in the model
        checkpoint.

    Each target is the bounded next state ``sz[t + 1]``, not an increment.
    """

    def __init__(
        self,
        sz: np.ndarray | torch.Tensor,
        particle_numbers: np.ndarray | torch.Tensor,
        time_end: float = 1.0,
        state_min: float = -1.0,
        state_max: float = 1.0,
        n_reference: float = 18.0,
    ) -> None:
        sz = torch.as_tensor(sz, dtype=torch.float32)
        particle_numbers = torch.as_tensor(particle_numbers, dtype=torch.float32)

        if sz.ndim != 2:
            raise ValueError(
                "sz must have shape (n_trajectories, n_time_points); "
                f"received {tuple(sz.shape)}."
            )
        if sz.shape[0] == 0 or sz.shape[1] < 2:
            raise ValueError("sz needs at least one trajectory with two time points.")
        if particle_numbers.ndim != 1 or particle_numbers.shape[0] != sz.shape[0]:
            raise ValueError(
                "particle_numbers must have shape (n_trajectories,); "
                f"received {tuple(particle_numbers.shape)} for {sz.shape[0]} "
                "trajectories."
            )
        if not torch.isfinite(sz).all():
            raise ValueError("sz contains NaN or infinite values.")
        if not torch.isfinite(particle_numbers).all() or torch.any(
            particle_numbers <= 0
        ):
            raise ValueError("particle_numbers must contain finite positive values.")
        if state_min >= state_max:
            raise ValueError("state_min must be smaller than state_max.")
        if time_end <= 0:
            raise ValueError("time_end must be positive.")
        if n_reference <= 0:
            raise ValueError("n_reference must be positive.")

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
        self.particle_numbers = particle_numbers.contiguous()
        self.normalized_particle_numbers = (
            self.particle_numbers / float(n_reference)
        )
        self.state_min = float(state_min)
        self.state_max = float(state_max)
        self.n_reference = float(n_reference)
        self.time_end = float(time_end)

        # Transition i starts at t_i, so use the first T - 1 times from a
        # T-point grid rather than stretching T - 1 times over the full range.
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
        current_sz = trajectory[:-1]
        n_feature = self.normalized_particle_numbers[index].expand_as(current_sz)
        features = torch.stack((current_sz, self.time, n_feature), dim=-1)
        return features, trajectory[1:]


def stratified_split_indices(
    particle_numbers: np.ndarray | torch.Tensor,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split trajectories independently within every available N value."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must lie in [0, 1).")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1.")

    values = torch.as_tensor(particle_numbers, dtype=torch.float32).cpu().numpy()
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("particle_numbers must be a non-empty one-dimensional array.")

    generator = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for particle_number in np.unique(values):
        indices = np.flatnonzero(values == particle_number)
        indices = generator.permutation(indices)

        n_train = int(train_fraction * len(indices))
        n_validation = int(validation_fraction * len(indices))
        n_test = len(indices) - n_train - n_validation
        if min(n_train, n_validation, n_test) < 1:
            raise ValueError(
                f"N={particle_number:g} has too few trajectories for the "
                "requested train/validation/test fractions."
            )

        train_indices.extend(indices[:n_train].tolist())
        validation_indices.extend(
            indices[n_train : n_train + n_validation].tolist()
        )
        test_indices.extend(indices[n_train + n_validation :].tolist())

    # Avoid batches ordered by N while remaining reproducible.
    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)
    generator.shuffle(test_indices)
    return train_indices, validation_indices, test_indices
