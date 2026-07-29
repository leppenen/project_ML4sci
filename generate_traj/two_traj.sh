#!/bin/bash
#
#PBS -N sz_traj_n4
#PBS -j oe
#PBS -q p72
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -J 500-999%30
#PBS -m eb
#PBS -M nikita.leppenen@weizmann.ac.il
#PBS -r y
#PBS -o out_sz_traj.txt

set -euo pipefail

date
hostname

eval "$('/apps01/apps/anaconda3-2022.10/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate nl97

script_dir="${PBS_O_WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$script_dir"

sample_id="${PBS_ARRAY_INDEX:-0}"
output_dir="${script_dir}/data/sz_N4"

python generate_sz_traj.py --sample-id "$sample_id" --output-dir "$output_dir"

date
