#!/bin/bash
#SBATCH -A antoniob
#SBATCH -q standby
#SBATCH --partition=a100-80gb
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:55:00
#SBATCH --job-name=wintertrain
#SBATCH --output=logs/%x_%j.out

# One ~4-hour training segment. Submitted (and chained) by submit_chain.sh.
#
# Usage (normally via sbatch): standby_train.sh <run_name> [train_net.py args...]
#
# Passes --auto-resume to train_net.py, so each segment continues the previous one's
# weights *and* LR schedule (loading the newest checkpoint + saved schedule/optimizer
# state from ../models/<run_name>/). This is a no-op on the first segment. A segment cut
# short by the 4h wall-clock limit is resumed from its last completed epoch.

set -euo pipefail

module load conda
conda activate /home/rosenth0/.conda/envs/cent7/2024.02-py311/chess_gfn

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/src"

RUN_NAME="${1:?Usage: standby_train.sh <run_name> [train args...]}"
shift

# Remaining arguments are passed straight through to train_net.py.
ARGS=(--name "$RUN_NAME" --auto-resume "$@")

echo "Running: python -u train_net.py ${ARGS[*]}"
python -u train_net.py "${ARGS[@]}"
