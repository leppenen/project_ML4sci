"""Dataset for trajectories conditioned on particle number and drive ratio."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiNOmegaNextStateDataset(Dataset):
    """Return ``[current_sz, normalized_time, normalized_N, normalized_omega]``."""

    def __init__(
        self,
        sz: np.ndarray | torch.Tensor,
        particle_numbers: np.ndarray | torch.Tensor,
        omega_ratios: np.ndarray | torch.Tensor,
        time_end: float = 1.0,
        state_min: float = -1.0,
        state_max: float = 1.0,
        n_reference: float = 18.0,
        omega_reference: float = 1.0,
    ) -> None:
        sz = torch.as_tensor(sz, dtype=torch.float32)
        particle_numbers = torch.as_tensor(particle_numbers, dtype=torch.float32)
        omega_ratios = torch.as_tensor(omega_ratios, dtype=torch.float32)
        if sz.ndim != 2 or sz.shape[0] == 0 or sz.shape[1] < 2:
            raise ValueError("sz must have shape (n_trajectories, n_time_points >= 2).")
        if particle_numbers.ndim != 1 or omega_ratios.ndim != 1:
            raise ValueError("conditioning values must be one-dimensional.")
        if particle_numbers.shape[0] != sz.shape[0] or omega_ratios.shape[0] != sz.shape[0]:
            raise ValueError("Each trajectory needs one N and one omega ratio.")
        if not torch.isfinite(sz).all() or not torch.isfinite(particle_numbers).all() or not torch.isfinite(omega_ratios).all():
            raise ValueError("Inputs contain NaN or infinite values.")
        if torch.any(particle_numbers <= 0) or torch.any(omega_ratios <= 0):
            raise ValueError("N and omega ratios must be positive.")
        if state_min >= state_max or time_end <= 0 or n_reference <= 0 or omega_reference <= 0:
            raise ValueError("Invalid state, time, N-reference, or Omega-reference value.")
        tolerance = 10 * torch.finfo(sz.dtype).eps
        if sz.min() < state_min - tolerance or sz.max() > state_max + tolerance:
            raise ValueError(f"sz lies outside [{state_min}, {state_max}].")

        self.sz = sz.contiguous()
        self.particle_numbers = particle_numbers.contiguous()
        self.omega_ratios = omega_ratios.contiguous()
        self.normalized_particle_numbers = self.particle_numbers / n_reference
        self.normalized_omega_ratios = self.omega_ratios / omega_reference
        self.state_min = float(state_min)
        self.state_max = float(state_max)
        self.n_reference = float(n_reference)
        self.omega_reference = float(omega_reference)
        self.time = torch.linspace(0.0, time_end, steps=sz.shape[1])[:-1]

    def __len__(self) -> int:
        return self.sz.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory = self.sz[index]
        features = torch.stack(
            (trajectory[:-1], self.time,
             self.normalized_particle_numbers[index].expand_as(trajectory[:-1]),
             self.normalized_omega_ratios[index].expand_as(trajectory[:-1])),
            dim=-1,
        )
        return features, trajectory[1:]


def stratified_split_indices(
    particle_numbers: np.ndarray | torch.Tensor,
    omega_ratios: np.ndarray | torch.Tensor,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split independently within every ``(N, omega_ratio)`` combination."""
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Invalid train/validation fractions.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1.")
    values_n = torch.as_tensor(particle_numbers).cpu().numpy()
    values_omega = torch.as_tensor(omega_ratios).cpu().numpy()
    if values_n.ndim != 1 or values_n.shape != values_omega.shape or len(values_n) == 0:
        raise ValueError("N and omega values must be non-empty, matching vectors.")

    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for key in sorted(set(zip(values_n.tolist(), values_omega.tolist()))):
        indices = np.flatnonzero((values_n == key[0]) & (values_omega == key[1]))
        indices = rng.permutation(indices)
        n_train = int(train_fraction * len(indices))
        n_validation = int(validation_fraction * len(indices))
        if min(n_train, n_validation, len(indices) - n_train - n_validation) < 1:
            raise ValueError(f"Condition {key} has too few trajectories for splitting.")
        train.extend(indices[:n_train].tolist())
        validation.extend(indices[n_train:n_train + n_validation].tolist())
        test.extend(indices[n_train + n_validation:].tolist())
    for values in (train, validation, test):
        rng.shuffle(values)
    return train, validation, test
