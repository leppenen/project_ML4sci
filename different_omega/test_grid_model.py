import numpy as np
import torch

from different_omega.dataset import MultiNOmegaNextStateDataset, stratified_split_indices
from different_omega.model import ConditionedNOmegaTrajectoryGRUGrid, interpolated_grid_nll
from different_omega.train_grid_model import generate_trajectories


def test_dataset_adds_n_and_omega_features():
    dataset = MultiNOmegaNextStateDataset(np.zeros((2, 4), dtype=np.float32), [4, 8], [.61, .71])
    features, target = dataset[0]
    assert features.shape == (3, 4)
    torch.testing.assert_close(features[:, 2], torch.full((3,), 4 / 18))
    torch.testing.assert_close(features[:, 3], torch.full((3,), .61))
    torch.testing.assert_close(target, torch.zeros(3))


def test_split_preserves_joint_conditions():
    n_values = np.repeat([4, 8], 20)
    omega_values = np.tile(np.repeat([.61, .71], 10), 2)
    splits = stratified_split_indices(n_values, omega_values, seed=3)
    for indices in splits:
        assert set(zip(n_values[indices], omega_values[indices])) == {(4, .61), (4, .71), (8, .61), (8, .71)}


def test_generation_accepts_batched_n_and_omega():
    model = ConditionedNOmegaTrajectoryGRUGrid(hidden_size=8, num_layers=1, grid_size=21)
    generated = generate_trajectories(model, [4, 8], [.61, .75], torch.full((2,), -1.0), 12)
    assert generated.shape == (2, 12)
    assert torch.all((generated >= -1) & (generated <= 1))
    assert torch.isfinite(interpolated_grid_nll(torch.zeros(2, 3), model(torch.zeros(2, 3, 4))[0]))
