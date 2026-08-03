"""Train one bounded trajectory model conditioned on particle number N.

Example on the cluster
----------------------
python withN_singleO/train_grid_model.py \
    --data-root GenerateTraj/data \
    --n-values 4 6 8 10 12 14 16 18 \
    --trajectories-per-n 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from .dataset import MultiNNextStateDataset, stratified_split_indices
    from .model import ConditionedTrajectoryGRUGrid, interpolated_grid_nll
except ImportError:
    # Support direct execution from the project root.
    from dataset import MultiNNextStateDataset, stratified_split_indices
    from model import ConditionedTrajectoryGRUGrid, interpolated_grid_nll


def default_data_root() -> Path:
    project_dir = Path(__file__).resolve().parent.parent
    candidates = [
        project_dir / "GenerateTraj" / "data",
        project_dir / "generate_traj" / "data",
    ]
    return next((path for path in candidates if path.is_dir()), candidates[0])


def load_trajectory_groups(
    data_root: Path,
    particle_numbers: Iterable[int],
    trajectories_per_n: int | None = 500,
) -> tuple[dict[int, np.ndarray], dict[int, list[str]]]:
    """Load balanced ``sz_N{N}/trajectory_*.npy`` groups."""
    groups: dict[int, np.ndarray] = {}
    source_files: dict[int, list[str]] = {}
    expected_time_points: int | None = None

    for particle_number in particle_numbers:
        directory = data_root / f"sz_N{particle_number}"
        files = sorted(directory.glob("trajectory_*.npy"))
        if not files:
            raise FileNotFoundError(
                f"No trajectory_*.npy files found in {directory.resolve()}."
            )
        if trajectories_per_n is not None:
            if len(files) < trajectories_per_n:
                raise ValueError(
                    f"N={particle_number} has {len(files)} trajectories, but "
                    f"{trajectories_per_n} were requested."
                )
            files = files[:trajectories_per_n]

        trajectories = [np.load(path, allow_pickle=False) for path in files]
        if any(trajectory.ndim != 1 for trajectory in trajectories):
            raise ValueError(f"Every trajectory for N={particle_number} must be 1-D.")

        array = np.stack(trajectories).astype(np.float32)
        if expected_time_points is None:
            expected_time_points = array.shape[1]
        elif array.shape[1] != expected_time_points:
            raise ValueError(
                "All N values must use the same time grid: expected "
                f"{expected_time_points} points, but N={particle_number} has "
                f"{array.shape[1]}."
            )

        groups[int(particle_number)] = array
        source_files[int(particle_number)] = [path.name for path in files]

    return groups, source_files


def combine_trajectory_groups(
    groups: dict[int, np.ndarray],
    particle_numbers: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate selected N groups and make one N label per trajectory."""
    selected = [int(value) for value in particle_numbers]
    if not selected:
        raise ValueError("At least one particle number is required.")

    trajectories = np.concatenate([groups[value] for value in selected], axis=0)
    labels = np.concatenate(
        [
            np.full(len(groups[value]), value, dtype=np.float32)
            for value in selected
        ]
    )
    return trajectories, labels


def evaluate(
    model: ConditionedTrajectoryGRUGrid,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_targets = 0

    with torch.no_grad():
        for features, next_sz in loader:
            features = features.to(device, non_blocking=True)
            next_sz = next_sz.to(device, non_blocking=True)
            logits, _ = model(features)
            loss = interpolated_grid_nll(
                next_sz,
                logits,
                state_min=model.state_min,
                state_max=model.state_max,
            )

            target_count = next_sz.numel()
            total_loss += loss.item() * target_count
            total_targets += target_count

    if total_targets == 0:
        raise ValueError("Cannot evaluate an empty loader.")
    return total_loss / total_targets


@torch.no_grad()
def generate_trajectories(
    model: ConditionedTrajectoryGRUGrid,
    particle_number: float | torch.Tensor,
    initial_sz: torch.Tensor,
    n_time_points: int,
    time_end: float = 1.0,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a batch conditioned on scalar or per-trajectory N values."""
    if n_time_points < 2:
        raise ValueError("n_time_points must be at least 2.")

    model.eval()
    device = next(model.parameters()).device
    current_sz = torch.as_tensor(
        initial_sz, dtype=torch.float32, device=device
    ).reshape(-1)
    if torch.any(current_sz < model.state_min) or torch.any(
        current_sz > model.state_max
    ):
        raise ValueError("initial_sz lies outside the model's state interval.")

    particle_number = torch.as_tensor(
        particle_number, dtype=torch.float32, device=device
    ).reshape(-1)
    if particle_number.numel() == 1:
        particle_number = particle_number.expand_as(current_sz)
    elif particle_number.shape != current_sz.shape:
        raise ValueError(
            "particle_number must be a scalar or contain one value per "
            "initial state."
        )
    normalized_n = model.encode_particle_number(particle_number)

    source_times = torch.linspace(
        0.0,
        time_end,
        steps=n_time_points,
        device=device,
    )[:-1]

    generated = [current_sz]
    hidden = None
    for time_value in source_times:
        time_feature = time_value.expand_as(current_sz)
        features = torch.stack(
            (current_sz, time_feature, normalized_n), dim=-1
        ).unsqueeze(1)
        logits, hidden = model(features, hidden)
        current_sz = model.sample_next_state(
            logits[:, 0],
            temperature=temperature,
            generator=generator,
        )
        generated.append(current_sz)

    return torch.stack(generated, dim=1)


def rollout_summary(generated: torch.Tensor, exact: torch.Tensor) -> dict[str, float]:
    """Compare free-running ensemble statistics with exact trajectories."""
    generated = generated.float()
    exact = exact.to(device=generated.device, dtype=generated.dtype)
    return {
        "generated_min": generated.min().item(),
        "generated_max": generated.max().item(),
        "mean_curve_mae": (
            generated.mean(dim=0) - exact.mean(dim=0)
        ).abs().mean().item(),
        "std_curve_mae": (
            generated.std(dim=0) - exact.std(dim=0)
        ).abs().mean().item(),
    }


def parse_args() -> argparse.Namespace:
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train one bounded grid model conditioned on particle number N."
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument(
        "--n-values", nargs="+", type=int, default=[4, 6, 8, 10, 12, 14, 16, 18]
    )
    parser.add_argument("--holdout-n", nargs="*", type=int, default=[])
    parser.add_argument("--trajectories-per-n", type=int, default=500)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=experiment_dir / "withN_grugrid_best.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=401)
    parser.add_argument("--state-min", type=float, default=-1.0)
    parser.add_argument("--state-max", type=float, default=1.0)
    parser.add_argument("--n-reference", type=float, default=18.0)
    parser.add_argument("--time-end", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if args.trajectories_per_n < 1:
        raise ValueError("trajectories_per_n must be positive.")

    all_n_values = sorted(set(args.n_values))
    holdout_n_values = sorted(set(args.holdout_n))
    unknown_holdouts = set(holdout_n_values) - set(all_n_values)
    if unknown_holdouts:
        raise ValueError(f"holdout N values are absent from --n-values: {unknown_holdouts}")
    fit_n_values = [value for value in all_n_values if value not in holdout_n_values]
    if not fit_n_values:
        raise ValueError("At least one N value must remain for training.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    groups, source_files = load_trajectory_groups(
        args.data_root.resolve(),
        all_n_values,
        trajectories_per_n=args.trajectories_per_n,
    )
    sz, particle_numbers = combine_trajectory_groups(groups, fit_n_values)
    dataset = MultiNNextStateDataset(
        sz,
        particle_numbers,
        time_end=args.time_end,
        state_min=args.state_min,
        state_max=args.state_max,
        n_reference=args.n_reference,
    )
    train_indices, validation_indices, test_indices = stratified_split_indices(
        particle_numbers, seed=args.seed
    )

    train_set = Subset(dataset, train_indices)
    validation_set = Subset(dataset, validation_indices)
    test_set = Subset(dataset, test_indices)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)

    model_config = {
        "input_size": 3,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "grid_size": args.grid_size,
        "dropout": args.dropout,
        "state_min": args.state_min,
        "state_max": args.state_max,
        "n_reference": args.n_reference,
    }
    model = ConditionedTrajectoryGRUGrid(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_validation_loss = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_targets = 0

        for features, next_sz in train_loader:
            features = features.to(device, non_blocking=True)
            next_sz = next_sz.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features)
            loss = interpolated_grid_nll(
                next_sz,
                logits,
                state_min=model.state_min,
                state_max=model.state_max,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            target_count = next_sz.numel()
            total_loss += loss.item() * target_count
            total_targets += target_count

        train_loss = total_loss / total_targets
        validation_loss = evaluate(model, validation_loader, device)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": "ConditionedTrajectoryGRUGrid",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_grid_nll": validation_loss,
                    "model_config": model_config,
                    "time_end": args.time_end,
                    "seed": args.seed,
                    "fit_n_values": fit_n_values,
                    "holdout_n_values": holdout_n_values,
                    "split_indices": {
                        "train": train_indices,
                        "validation": validation_indices,
                        "test": test_indices,
                    },
                    "source_files": {
                        str(value): source_files[value] for value in all_n_values
                    },
                },
                args.checkpoint,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train grid NLL: {train_loss:.5f} | "
                f"validation grid NLL: {validation_loss:.5f}"
            )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_loader, device)
    print(f"Using: {device}")
    print(f"Fit N values: {fit_n_values}")
    print(f"Held-out N values: {holdout_n_values}")
    print(f"Best checkpoint epoch: {checkpoint['epoch']}")
    print(f"Seen-N test grid NLL: {test_loss:.5f}")
    print(f"Saved checkpoint: {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
