# Trajectory GRU Grid Model for Open Quantum Systems 

This repository contains a PyTorch-based deep learning framework for learning and generating stochastic quantum trajectories. The project builds on my PhD research in open quantum systems, combining Python QuTiP-based Monte Carlo wave-function (MCWF) simulations with recurrent neural networks (GRUs).

The physical model consists of a driven array of quantum emitters in a cavity with finite cooperativity, which accounts for both collective and local dissipation. By training on bounded state grids generated via QuTiP, the GRU learns to emulate the time evolution of collective spin observables, such as magnetization ($S_z$), and predicts new trajectories in computationally inaccessible regimes. 

## Background & Motivation

In our recent paper, [Leppenen & Shahmoon, Phys. Rev. A **114**, 013720 (2026)](https://doi.org/10.1103/cy7b-hsl3), we demonstrated that for certain drive values, the system exhibits switching between two quantum states—a phenomenon directly observable in single quantum trajectories. 

Due to the exponential growth of the Hilbert space, exact quantum trajectory simulations are currently restricted to system sizes up to $N = 18$. The goal of this machine learning model is to learn the magnetization dynamics over time from these simulated trajectories and predict the system's behavior for larger $N$, as well as across different laser drive regions.

The motivation slides are located in `tex_notes/`. See `prpsl.pdf`.


## Overview

The codebase is organized around a workflow:

1. Generate stochastic quantum trajectories with QuTiP.
2. Convert them into training datasets.
3. Train a GRU-based probabilistic model on the trajectory data.
4. Generate new trajectories by autoregressive sampling.

This project includes multiple model variants for different conditioning setups:

- `att_N8/` and `att_N18/`: models trained for a fixed particle number `N`.
- `withN_singleO/`: model conditioned on particle number `N` for a fixed drive ratio.
- `different_omega/`: model conditioned on both `N` and `omega / omega_c`.
- `generate_traj/`: tools to generate and package raw trajectory data.

---

## Results

The main findings of this project are summarized in the poster [`Poster_TrajGRU.pdf`](Poster_TrajGRU.pdf), which was presented at the project defense for the *Deep Learning for Science* course at the Weizmann Institute of Science.


## Requirements

This project expects a Python environment with:

- Python 3.10+
- NumPy
- SciPy
- PyTorch
- QuTiP

A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy torch qutip
```

If you want a more reproducible environment, you can also use conda or mamba.

---

## Data generation workflow

The raw stochastic trajectories are generated with QuTiP using Monte Carlo wavefunction simulations.

### Generate a single trajectory

```bash
python generate_traj/generate_sz_traj.py \
  --sample-id 0 \
  --output-dir generate_traj/data/sz_N4
```

This writes a `.npy` file containing the trajectory values for $S_z(t)$.

### Build a dataset from many trajectories

```bash
python generate_traj/build_dataset.py \
  --input-dir generate_traj/data/sz_N4 \
  --output-dir generate_traj/data/processed \
  --N 4 \
  --num-trajectories 1000 \
  --first-id 0
```

The script verifies file consistency and saves a compressed NumPy dataset plus train/validation/test split indices.

---

## Training the models

### Fixed-size model for N=8

```bash
python att_N8/train_grid_model.py \
  --data-dir generate_traj/data/sz_N8 \
  --epochs 100 \
  --batch-size 64 \
  --hidden-size 128 \
  --grid-size 401
```

### Fixed-size model for N=18

```bash
python att_N18/train_grid_model.py \
  --data-dir generate_traj/data/sz_N18 \
  --epochs 100 \
  --batch-size 64 \
  --hidden-size 128 \
  --grid-size 401
```

### Conditioned model on N and drive ratio

```bash
python different_omega/train_grid_model.py \
  --data-root generate_traj/data \
  --n-values 4 6 8 10 12 14 16 18 \
  --omega-values 0.73 \
  --epochs 100 \
  --batch-size 64
```

The training scripts save best-performing checkpoints as `.pt` files and print validation losses during optimization.

---

## Model architecture

The core idea is a recurrent probabilistic model:

- A GRU processes the trajectory history.
- The hidden state encodes temporal memory.
- A final head emits logits over a bounded state grid.
- A softmax-based categorical likelihood is used for the next state.
- The loss is computed using a grid-based interpolated negative log-likelihood.

This gives the model a way to learn stochastic dynamics while respecting a bounded physical state range.

---

## Acknowledgements

This project sits at the intersection of:

- quantum trajectory simulation with QuTiP,
- time-series modeling with recurrent neural networks,
- and generative modeling of stochastic observables in dissipative quantum systems.

If you use the code in research, please cite the relevant project or include a note describing the parameter regime and training setup.
