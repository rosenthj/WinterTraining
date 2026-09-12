#!/bin/bash
#SBATCH -A antoniob
#SBATCH -q standby
#SBATCH --partition=a100-80gb
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:55:00
#SBATCH --job-name=wintertrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --signal=USR1@180

# One ~4-hour training segment. Submitted (and chained) by submit_chain.sh.
#
# Usage (normally via sbatch): standby_train.sh <run_name> [train_net.py args...]
#
# Passes --auto-resume to train_net.py, so each segment continues the previous one's
# weights *and* LR schedule (loading the checkpoint + saved schedule/optimizer state from
# ../models/<run_name>/). This is a no-op on the first segment.
#
# Ending a segment without losing work. A segment dies for one of three reasons, and the
# trainer handles all three at *batch* granularity rather than epoch granularity, so a
# boundary costs at most one checkpoint interval instead of every batch since the last
# epoch ended:
#
#  1. The wall-clock limit. train_net.py reads SLURM_JOB_END_TIME and, --stop-margin-secs
#     before it, checkpoints mid-epoch and exits 0. This is the primary mechanism: it is
#     authoritative and needs nothing from the scheduler.
#  2. A signal. --signal=USR1@180 above asks SLURM to send SIGUSR1 three minutes before the
#     limit, and a standby preemption sends SIGTERM within its grace period; either one makes
#     the trainer checkpoint and exit at the next batch boundary. Backup for (1), and the
#     only warning available when a segment ends early.
#  3. No warning at all (preemption past its grace period, node failure). Nothing can be
#     caught here, so --checkpoint-every-mins bounds the loss to that interval.
#
# The trap keeps the batch shell alive for (2): bash defers a trapped signal until the
# foreground command returns, so it waits for python to finish checkpointing instead of
# exiting immediately and letting slurmstepd tear the step down underneath it.

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# Machine-specific paths live in the git-ignored local_env.sh; see local_env.sh.example.
if [ -f "$REPO_DIR/local_env.sh" ]; then
    # shellcheck source=/dev/null
    source "$REPO_DIR/local_env.sh"
fi
: "${WINTER_CONDA_ENV:?set WINTER_CONDA_ENV (copy local_env.sh.example to local_env.sh)}"

trap 'echo "Signal received; the trainer checkpoints and exits at its next batch boundary."' USR1 TERM

module load conda
conda activate "$WINTER_CONDA_ENV"

cd "$REPO_DIR/src"

RUN_NAME="${1:?Usage: standby_train.sh <run_name> [train args...]}"
shift

# Remaining arguments are passed straight through to train_net.py.
ARGS=(--name "$RUN_NAME" --auto-resume "$@")

echo "Running: python -u train_net.py ${ARGS[*]}"
python -u train_net.py "${ARGS[@]}"
