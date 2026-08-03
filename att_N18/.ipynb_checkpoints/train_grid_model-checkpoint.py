"""Train and evaluate a bounded categorical trajectory model for N=18.

Example
-------
python att_N18/train_grid_model.py \
    --data-dir generate_traj/data/sz_N8 \
    --grid-size 401
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

try:
    from .dataset import SingleNNextStateDataset
    from .model import TrajectoryGRUGrid, interpolated_grid_nll
except ImportError:
    # Support direct execution: python att_N8/train_grid_model.py
    from dataset import SingleNNextStateDataset
    from model import TrajectoryGRUGrid, interpolated_grid_nll


def parse_args() -> argparse.Namespace:
    experiment_dir = Path(__file__).resolve().parent
    project_dir = experiment_dir.parent

    parser = argparse.ArgumentParser(
        description="Train a GRU categorical model on a bounded state grid."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "generate_traj" / "data" / "sz_N18",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=experiment_dir / "n18_grugrid_best.pt",
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
    parser.add_argument("--time-end", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def evaluate(
    model: TrajectoryGRUGrid,
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

    return total_loss / total_targets


@torch.no_grad()
def generate_trajectories(
    model: TrajectoryGRUGrid,
    initial_sz: torch.Tensor,
    n_time_points: int,
    time_end: float = 1.0,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a batch with shape ``(batch, n_time_points)``."""
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

    # These are source-state times t_0, ..., t_(T-2).
    normalized_times = torch.linspace(
        0.0,
        time_end,
        steps=n_time_points,
        device=device,
    )[:-1]

    generated = [current_sz]
    hidden = None

    for time_value in normalized_times:
        time_feature = time_value.expand_as(current_sz)
        features = torch.stack((current_sz, time_feature), dim=-1).unsqueeze(1)
        logits, hidden = model(features, hidden)
        current_sz = model.sample_next_state(
            logits[:, 0],
            temperature=temperature,
            generator=generator,
        )
        generated.append(current_sz)

    return torch.stack(generated, dim=1)


def rollout_summary(generated: torch.Tensor, exact: torch.Tensor) -> dict[str, float]:
    """Compare free-running ensemble statistics with held-out trajectories."""
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


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(args.data_dir.resolve().glob("trajectory_*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No trajectory_*.npy files found in {args.data_dir.resolve()}."
        )

    sz = np.stack([np.load(path) for path in files]).astype(np.float32)
    dataset = SingleNNextStateDataset(
        sz,
        time_end=args.time_end,
        state_min=args.state_min,
        state_max=args.state_max,
    )

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("The dataset is too small for an 80/10/10 split.")

    split_generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=split_generator,
    )

    pin_memory = device.type == "cuda"
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=shuffle_generator,
        **loader_options,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)

    model_config = {
        "input_size": 2,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "grid_size": args.grid_size,
        "dropout": args.dropout,
        "state_min": args.state_min,
        "state_max": args.state_max,
    }
    model = TrajectoryGRUGrid(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
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
        val_loss = evaluate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": "TrajectoryGRUGrid",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_grid_nll": val_loss,
                    "model_config": model_config,
                    "time_end": args.time_end,
                    "seed": args.seed,
                    "split_indices": {
                        "train": list(train_set.indices),
                        "validation": list(val_set.indices),
                        "test": list(test_set.indices),
                    },
                    "source_files": [path.name for path in files],
                },
                args.checkpoint,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train grid NLL: {train_loss:.5f} | "
                f"validation grid NLL: {val_loss:.5f}"
            )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_loader, device)

    test_indices = torch.as_tensor(test_set.indices, dtype=torch.long)
    exact_test = dataset.sz[test_indices].to(device)
    rollout_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    generated_test = generate_trajectories(
        model,
        initial_sz=exact_test[:, 0],
        n_time_points=exact_test.shape[1],
        time_end=args.time_end,
        generator=rollout_generator,
    )
    summary = rollout_summary(generated_test, exact_test)

    print(f"Using: {device}")
    print(f"Best checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test grid NLL: {test_loss:.5f}")
    print(
        "Generated range: "
        f"[{summary['generated_min']:.6f}, {summary['generated_max']:.6f}]"
    )
    print(f"Mean-curve MAE: {summary['mean_curve_mae']:.6f}")
    print(f"Std-curve MAE: {summary['std_curve_mae']:.6f}")
    print(f"Saved checkpoint: {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
