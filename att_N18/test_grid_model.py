"""Focused tests for the bounded grid trajectory model."""

import numpy as np
import torch

from att_N18.dataset import SingleNNextStateDataset
from att_N18.model import TrajectoryGRUGrid, interpolated_grid_nll
from att_N18.train_grid_model import generate_trajectories


def test_dataset_returns_next_state_and_source_time_grid() -> None:
    sz = np.array([[-1.0, -0.5, 0.25, 1.0]], dtype=np.float32)
    dataset = SingleNNextStateDataset(sz)

    features, target = dataset[0]

    torch.testing.assert_close(features[:, 0], torch.tensor([-1.0, -0.5, 0.25]))
    torch.testing.assert_close(features[:, 1], torch.tensor([0.0, 1 / 3, 2 / 3]))
    torch.testing.assert_close(target, torch.tensor([-0.5, 0.25, 1.0]))


def test_interpolated_loss_uses_bracketing_grid_points() -> None:
    # On grid [-1, 0, 1], target 0.25 has weights 0.75 at 0 and 0.25 at 1.
    logits = torch.log(torch.tensor([[[0.1, 0.6, 0.3]]]))
    target = torch.tensor([[0.25]])

    loss = interpolated_grid_nll(target, logits)
    expected = -(0.75 * torch.log(torch.tensor(0.6)) + 0.25 * torch.log(torch.tensor(0.3)))

    torch.testing.assert_close(loss, expected)


def test_generated_trajectories_stay_in_physical_interval() -> None:
    torch.manual_seed(4)
    model = TrajectoryGRUGrid(
        hidden_size=8,
        num_layers=1,
        grid_size=21,
        state_min=-1.0,
        state_max=1.0,
    )

    generated = generate_trajectories(
        model,
        initial_sz=torch.tensor([-1.0, 0.0, 1.0]),
        n_time_points=25,
    )

    assert generated.shape == (3, 25)
    assert torch.all(generated >= -1.0)
    assert torch.all(generated <= 1.0)
