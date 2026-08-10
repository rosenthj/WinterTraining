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
# on the cluster. This is CPU-only work (python-chess + scipy) and the requested GPU sits
# idle throughout, but a GPU-less job cannot be submitted to these partitions, and the A100
# partitions have far more nodes than the others -- so asking for an A100 actually schedules
# sooner than asking for a lesser GPU.
#
# Sizing: a shard of ~40k games converts in minutes, where a 475k-game file needs roughly
# 1.8 hours of local-equivalent work against the 3:55 window -- thin enough that a slower
# node alone exhausts it. Prefer converting per shard (see the datagen pipeline) over
# feeding whole merged PGNs through this script.
#
# Usage (normally via sbatch, fanned out by submit_gen.sh):
#   generate_dataset.sh <name> [extra pgn_to_dataset.py args...]
#
# <name> is the shared PGN/dataset base name: e.g. desk_v300 reads ../pgns/desk_v300.pgn
# and writes ../datasets/features_desk_v300.npz and ../datasets/targets_desk_v300.npz.

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# Machine-specific paths live in the git-ignored local_env.sh; see local_env.sh.example.
# Anything already exported wins, so a one-off run can override without editing it.
if [ -f "$REPO_DIR/local_env.sh" ]; then
    # shellcheck source=/dev/null
    source "$REPO_DIR/local_env.sh"
fi
: "${WINTER_CONDA_ENV:?set WINTER_CONDA_ENV (copy local_env.sh.example to local_env.sh)}"
: "${WINTER_TB_PATH:?set WINTER_TB_PATH (copy local_env.sh.example to local_env.sh)}"

module load conda
conda activate "$WINTER_CONDA_ENV"

cd "$REPO_DIR/src"

NAME="${1:?Usage: generate_dataset.sh <name> [pgn_to_dataset.py args...]}"
shift

TB_PATH="${TB_PATH:-$WINTER_TB_PATH}"

ARGS=(--name "$NAME" --tablebase "$TB_PATH" "$@")
echo "Running: python -u pgn_to_dataset.py ${ARGS[*]}"
python -u pgn_to_dataset.py "${ARGS[@]}"
