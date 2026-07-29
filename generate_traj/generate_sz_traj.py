#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from qutip import Options, basis, mcsolve, qeye, sigmam, sigmax, sigmaz, tensor


N = 4
G1D = 10.0
G0 = 1.0
WC = G1D * (N - 1) / 4.0
WX = 0.73 * WC
TLIST = np.linspace(0.0, 70.0, 210)


def build_chain_operator(single_site_operator, site_index: int):
    operators = [qeye(2) for _ in range(N)]
    operators[site_index] = single_site_operator
    return tensor(operators)


def collective_operator(single_site_operator):
    operator = build_chain_operator(single_site_operator, 0)
    for site_index in range(1, N):
        operator = operator + build_chain_operator(single_site_operator, site_index)
    return operator


def build_model():
    sz = collective_operator(sigmaz())
    sx = collective_operator(sigmax())
    sm = collective_operator(sigmam())

    ham_mc = WX * sx
    c_op_list = [np.sqrt(G0) * build_chain_operator(sigmam(), site_index) for site_index in range(N)]
    c_op_list.append(np.sqrt(G1D) * sm)

    psi0 = tensor([basis(2, 1) for _ in range(N)])
    return ham_mc, c_op_list, psi0, sz


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one stochastic Sz trajectory.")
    parser.add_argument("--sample-id", type=int, default=0, help="Trajectory index used for the output filename.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where the trajectory will be saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ham_mc, c_op_list, psi0, sz_op = build_model()
    result = mcsolve(
        ham_mc,
        psi0,
        TLIST,
        c_op_list,
        [sz_op],
        ntraj=1,
        options=Options(num_cpus=1),
    )

    sz_t = np.asarray(result.expect[0], dtype=np.float32) / N
    output_path = args.output_dir / f"trajectory_{args.sample_id:04d}.npy"
    np.save(output_path, sz_t)


if __name__ == "__main__":
    main()
