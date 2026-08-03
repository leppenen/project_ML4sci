"""Tests for the particle-number-conditioned bounded grid model."""

import numpy as np
import torch

from withN_singleO.dataset import MultiNNextStateDataset, stratified_split_indices
from withN_singleO.model import (
    ConditionedTrajectoryGRUGrid,
    interpolated_grid_nll,
)
from withN_singleO.train_grid_model import generate_trajectories


def test_dataset_adds_normalized_particle_number() -> None:
    sz = np.array(
        [
            [-1.0, -0.5, 0.0, 0.5],
            [-1.0, -0.75, -0.5, -0.25],
        ],
        dtype=np.float32,
    )
    dataset = MultiNNextStateDataset(sz, [4, 18], n_reference=18)

    features_n4, target_n4 = dataset[0]
    features_n18, _ = dataset[1]

    assert features_n4.shape == (3, 3)
    torch.testing.assert_close(features_n4[:, 0], torch.tensor([-1.0, -0.5, 0.0]))
    torch.testing.assert_close(features_n4[:, 1], torch.tensor([0.0, 1 / 3, 2 / 3]))
    torch.testing.assert_close(features_n4[:, 2], torch.full((3,), 4 / 18))
    torch.testing.assert_close(features_n18[:, 2], torch.ones(3))
    torch.testing.assert_close(target_n4, torch.tensor([-0.5, 0.0, 0.5]))


def test_split_contains_every_particle_number() -> None:
    particle_numbers = np.repeat([4, 6, 8], 10)
    train, validation, test = stratified_split_indices(particle_numbers, seed=7)

    assert len(train) == 24
    assert len(validation) == 3
    assert len(test) == 3
    for indices in (train, validation, test):
        assert set(particle_numbers[indices]) == {4, 6, 8}


def test_model_and_interpolated_loss_shapes() -> None:
    model = ConditionedTrajectoryGRUGrid(
        hidden_size=8,
        num_layers=1,
        grid_size=21,
    )
    features = torch.randn(3, 7, 3)
    target = torch.zeros(3, 7)

    logits, hidden = model(features)
    loss = interpolated_grid_nll(target, logits)

    assert logits.shape == (3, 7, 21)
    assert hidden.shape == (1, 3, 8)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_generation_accepts_extrapolated_n_and_remains_bounded() -> None:
    torch.manual_seed(4)
    model = ConditionedTrajectoryGRUGrid(
        hidden_size=8,
        num_layers=1,
        grid_size=21,
        n_reference=18,
    )
    generated = generate_trajectories(
        model,
        particle_number=24,
        initial_sz=torch.full((5,), -1.0),
        n_time_points=25,
    )

    assert generated.shape == (5, 25)
    assert torch.all(generated >= -1.0)
    assert torch.all(generated <= 1.0)
