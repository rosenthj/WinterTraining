#!/bin/bash
#SBATCH -A antoniob
#SBATCH -q standby
#SBATCH --partition=a100-80gb
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:55:00
#SBATCH --job-name=wintergen
#SBATCH --output=logs/%x_%j.out

# Convert one PGN file into a training dataset (features_<name>.npz + targets_<name>.npz)
# on the cluster. This is CPU-only work (python-chess + scipy); the GPU requested by the
# a100-80gb partition sits idle. If your partition allows GPU-less jobs you can drop the
# `--gpus-per-node=1` line above to schedule faster and free the GPU. After the
# get_features optimizations a ~250k-game file converts comfortably within the 3:55 window.
#
# Usage (normally via sbatch, fanned out by submit_gen.sh):
#   generate_dataset.sh <name> [extra pgn_to_dataset.py args...]
#
# <name> is the shared PGN/dataset base name: e.g. desk_v300 reads ../pgns/desk_v300.pgn
# and writes ../datasets/features_desk_v300.npz and ../datasets/targets_desk_v300.npz.

set -euo pipefail

module load conda
conda activate /home/rosenth0/.conda/envs/cent7/2024.02-py311/chess_gfn

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/src"

NAME="${1:?Usage: generate_dataset.sh <name> [pgn_to_dataset.py args...]}"
shift

# Syzygy tablebase location on the cluster. Override by exporting TB_PATH before sbatch.
TB_PATH="${TB_PATH:-/scratch/gilbreth/rosenth0/data/egtb6/}"

ARGS=(--name "$NAME" --tablebase "$TB_PATH" "$@")
echo "Running: python -u pgn_to_dataset.py ${ARGS[*]}"
python -u pgn_to_dataset.py "${ARGS[@]}"
