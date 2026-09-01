"""Train a bounded GRU-grid model over N and Omega/Omega_c."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from .dataset import MultiNOmegaNextStateDataset, stratified_split_indices
    from .model import ConditionedNOmegaTrajectoryGRUGrid, interpolated_grid_nll
except ImportError:
    from dataset import MultiNOmegaNextStateDataset, stratified_split_indices
    from model import ConditionedNOmegaTrajectoryGRUGrid, interpolated_grid_nll


def default_data_root() -> Path:
    return Path(__file__).resolve().parent.parent / "GenerateTraj"


def omega_from_directory(directory: Path) -> float:
    if directory.name == "data":
        return 0.73
    suffix = directory.name.removeprefix("data_")
    if not suffix.isdigit():
        raise ValueError(f"Cannot infer Omega/Omega_c from {directory.name}.")
    return int(suffix) / 100.0


def load_trajectories(data_root: Path, n_values: list[int], omega_values: list[float] | None,
                      trajectories_per_condition: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    roots = sorted(data_root.glob("data*"))
    if not roots:
        raise FileNotFoundError(f"No data* folders found in {data_root.resolve()}.")
    selected = []
    for root in roots:
        omega = omega_from_directory(root)
        if omega_values is None or any(np.isclose(omega, value) for value in omega_values):
            selected.append((root, omega))
    if not selected:
        raise ValueError("No data* folder matches --omega-values.")

    trajectories, labels_n, labels_omega, source_files = [], [], [], {}
    time_points = None
    for root, omega in selected:
        for particle_number in n_values:
            directory = root / f"sz_N{particle_number}"
            files = sorted(directory.glob("trajectory_*.npy"))
            if not files:
                continue
            if trajectories_per_condition is not None:
                files = files[:trajectories_per_condition]
            arrays = [np.asarray(np.load(path, allow_pickle=False), dtype=np.float32) for path in files]
            if any(array.ndim != 1 for array in arrays):
                raise ValueError(f"Trajectories in {directory} must be one-dimensional.")
            current = np.stack(arrays)
            if time_points is None:
                time_points = current.shape[1]
            elif current.shape[1] != time_points:
                raise ValueError("All conditions must use the same number of time points.")
            trajectories.append(current)
            labels_n.append(np.full(len(files), particle_number, dtype=np.float32))
            labels_omega.append(np.full(len(files), omega, dtype=np.float32))
            source_files[f"{omega:g}/N{particle_number}"] = [path.name for path in files]
    if not trajectories:
        raise FileNotFoundError(f"No requested trajectories found below {data_root.resolve()}.")
    return np.concatenate(trajectories), np.concatenate(labels_n), np.concatenate(labels_omega), source_files


def generate_trajectories(model, particle_number, omega_ratio, initial_sz, n_time_points,
                          time_end=1.0, temperature=1.0, generator=None):
    if n_time_points < 2:
        raise ValueError("n_time_points must be at least 2.")
    device = next(model.parameters()).device
    current = torch.as_tensor(initial_sz, dtype=torch.float32, device=device).reshape(-1)
    particle_number = torch.as_tensor(particle_number, dtype=torch.float32, device=device).reshape(-1)
    omega_ratio = torch.as_tensor(omega_ratio, dtype=torch.float32, device=device).reshape(-1)
    if particle_number.numel() == 1:
        particle_number = particle_number.expand_as(current)
    if omega_ratio.numel() == 1:
        omega_ratio = omega_ratio.expand_as(current)
    if particle_number.shape != current.shape or omega_ratio.shape != current.shape:
        raise ValueError("N, omega_ratio, and initial_sz must be scalar or matching batches.")
    if torch.any(current < model.state_min) or torch.any(current > model.state_max):
        raise ValueError("initial_sz lies outside the model state interval.")
    times = torch.linspace(0.0, time_end, n_time_points, device=device)[:-1]
    normalized_n = model.encode_particle_number(particle_number)
    normalized_omega = model.encode_omega_ratio(omega_ratio)
    generated, hidden = [current], None
    model.eval()
    with torch.no_grad():
        for time_value in times:
            features = torch.stack((current, time_value.expand_as(current), normalized_n, normalized_omega), dim=-1).unsqueeze(1)
            logits, hidden = model(features, hidden)
            current = model.sample_next_state(logits[:, 0], temperature, generator)
            generated.append(current)
    return torch.stack(generated, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--n-values", nargs="+", type=int, default=[4, 6, 8, 10, 12, 14, 16, 18])
    parser.add_argument("--omega-values", nargs="*", type=float, default=None)
    parser.add_argument("--trajectories-per-condition", type=int, default=500)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).with_name("different_omega_grugrid_best.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=401)
    parser.add_argument("--n-reference", type=float, default=18.0)
    parser.add_argument("--omega-reference", type=float, default=1.0)
    parser.add_argument("--time-end", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sz, values_n, values_omega, source_files = load_trajectories(args.data_root, args.n_values, args.omega_values, args.trajectories_per_condition)
    dataset = MultiNOmegaNextStateDataset(sz, values_n, values_omega, time_end=args.time_end, n_reference=args.n_reference, omega_reference=args.omega_reference)
    train_i, validation_i, test_i = stratified_split_indices(values_n, values_omega, seed=args.seed)
    options = dict(batch_size=args.batch_size, num_workers=0, pin_memory=device.type == "cuda")
    train_loader = DataLoader(Subset(dataset, train_i), shuffle=True, **options)
    validation_loader = DataLoader(Subset(dataset, validation_i), shuffle=False, **options)
    test_loader = DataLoader(Subset(dataset, test_i), shuffle=False, **options)
    config = dict(input_size=4, hidden_size=args.hidden_size, num_layers=args.num_layers, grid_size=args.grid_size, dropout=args.dropout, n_reference=args.n_reference, omega_reference=args.omega_reference)
    model = ConditionedNOmegaTrajectoryGRUGrid(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = interpolated_grid_nll(target, model(features)[0], model.state_min, model.state_max)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = np.mean([interpolated_grid_nll(target.to(device), model(features.to(device))[0], model.state_min, model.state_max).item() for features, target in validation_loader])
        if validation_loss < best:
            best = validation_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "model_config": config, "time_end": args.time_end, "n_values": sorted(set(values_n.tolist())), "omega_values": sorted(set(values_omega.tolist())), "source_files": source_files}, args.checkpoint)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | validation grid NLL: {validation_loss:.5f}")
    print(f"Using: {device}; saved checkpoint: {args.checkpoint.resolve()}")
    print(f"Training conditions: N={sorted(set(values_n.tolist()))}, Omega/Omega_c={sorted(set(values_omega.tolist()))}")


if __name__ == "__main__":
    main()
