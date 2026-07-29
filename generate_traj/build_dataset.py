#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


TRAJECTORY_PATTERN = re.compile(r"trajectory_(\d+)\.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine individual Sz trajectories into a training dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/sz_N4"),
        help="Directory containing trajectory_XXXX.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for the processed dataset and split files.",
    )
    parser.add_argument("--num-trajectories", type=int, default=1000)
    parser.add_argument("--first-id", type=int, default=0)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--omega-ratio", type=float, default=0.73)
    parser.add_argument("--gamma-ratio", type=float, default=0.1)
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-stop", type=float, default=70.0)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=2026,
        help="Seed used only to make the train/validation/test split reproducible.",
    )
    return parser.parse_args()


def find_trajectory_files(
    input_dir: Path, first_id: int, num_trajectories: int
) -> list[tuple[int, Path]]:
    files_by_id: dict[int, Path] = {}

    for path in input_dir.glob("trajectory_*.npy"):
        match = TRAJECTORY_PATTERN.fullmatch(path.name)
        if match is None:
            continue

        trajectory_id = int(match.group(1))
        if trajectory_id in files_by_id:
            raise ValueError(f"Duplicate trajectory ID {trajectory_id}")
        files_by_id[trajectory_id] = path

    expected_ids = set(range(first_id, first_id + num_trajectories))
    found_ids = set(files_by_id)
    missing_ids = sorted(expected_ids - found_ids)
    unexpected_ids = sorted(found_ids - expected_ids)

    if missing_ids:
        preview = ", ".join(map(str, missing_ids[:10]))
        suffix = " ..." if len(missing_ids) > 10 else ""
        raise FileNotFoundError(f"Missing trajectory IDs: {preview}{suffix}")
    if unexpected_ids:
        preview = ", ".join(map(str, unexpected_ids[:10]))
        suffix = " ..." if len(unexpected_ids) > 10 else ""
        raise ValueError(f"Unexpected trajectory IDs: {preview}{suffix}")

    return [(trajectory_id, files_by_id[trajectory_id]) for trajectory_id in sorted(expected_ids)]


def load_trajectories(files: list[tuple[int, Path]]) -> tuple[np.ndarray, np.ndarray]:
    trajectory_ids: list[int] = []
    trajectories: list[np.ndarray] = []
    sequence_length: int | None = None

    for trajectory_id, path in files:
        trajectory = np.load(path, allow_pickle=False)

        if trajectory.ndim != 1:
            raise ValueError(f"{path} has shape {trajectory.shape}; expected a 1D array")
        if sequence_length is None:
            sequence_length = len(trajectory)
        elif len(trajectory) != sequence_length:
            raise ValueError(
                f"{path} has {len(trajectory)} time points; expected {sequence_length}"
            )
        if not np.isfinite(trajectory).all():
            raise ValueError(f"{path} contains NaN or infinite values")

        trajectory_ids.append(trajectory_id)
        trajectories.append(np.asarray(trajectory, dtype=np.float32))

    return np.stack(trajectories), np.asarray(trajectory_ids, dtype=np.int32)


def make_splits(
    num_trajectories: int, split_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.random.default_rng(split_seed).permutation(num_trajectories)
    num_train = int(0.70 * num_trajectories)
    num_validation = int(0.15 * num_trajectories)

    train_indices = indices[:num_train]
    validation_indices = indices[num_train : num_train + num_validation]
    test_indices = indices[num_train + num_validation :]
    return train_indices, validation_indices, test_indices


def main() -> None:
    args = parse_args()
    files = find_trajectory_files(
        args.input_dir, args.first_id, args.num_trajectories
    )
    sz, trajectory_ids = load_trajectories(files)

    num_trajectories, num_time_points = sz.shape
    system_sizes = np.full(num_trajectories, args.N, dtype=np.int16)
    omega_ratios = np.full(num_trajectories, args.omega_ratio, dtype=np.float32)
    gamma_ratios = np.full(num_trajectories, args.gamma_ratio, dtype=np.float32)
    times = np.linspace(
        args.time_start, args.time_stop, num_time_points, dtype=np.float32
    )

    train_indices, validation_indices, test_indices = make_splits(
        num_trajectories, args.split_seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / f"trajectories_N{args.N}.npz"
    splits_path = args.output_dir / f"splits_N{args.N}.npz"

    np.savez_compressed(
        dataset_path,
        sz=sz,
        trajectory_id=trajectory_ids,
        t=times,
        N=system_sizes,
        omega_ratio=omega_ratios,
        gamma_ratio=gamma_ratios,
    )
    np.savez(
        splits_path,
        train=train_indices,
        validation=validation_indices,
        test=test_indices,
    )

    print(f"Saved dataset: {dataset_path}")
    print(f"  sz shape: {sz.shape}")
    print(f"Saved splits: {splits_path}")
    print(
        "  train/validation/test: "
        f"{len(train_indices)}/{len(validation_indices)}/{len(test_indices)}"
    )


if __name__ == "__main__":
    main()
