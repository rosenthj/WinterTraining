#!/bin/bash
#
# Enqueue a training run as a chain of ~4-hour SLURM segments. Each segment is held
# until the previous one finishes (afterany: succeeds, fails, or hits the time limit),
# so a single long training run survives the standby queue's 4h-per-job limit.
#
# Usage:
#   ./submit_chain.sh <num_segments> <run_name> [train_net.py args...]
#
# Examples:
#   ./submit_chain.sh 6 baseline --datasets all --exclude vEnd --batch-size 256
#   ./submit_chain.sh 3 frc_v2  --datasets 200-221 --init-lr 0.008 --min-lr 0.0001
#
# Each segment runs:  standby_train.sh <run_name> [train args...]
# which auto-resumes from the newest checkpoint in ../models/<run_name>/.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <num_segments> <run_name> [train_net.py args...]" >&2
    exit 1
fi

NUM_SEGMENTS="$1"
RUN_NAME="$2"
shift 2

if ! [[ "$NUM_SEGMENTS" =~ ^[0-9]+$ ]] || [ "$NUM_SEGMENTS" -lt 1 ]; then
    echo "Error: num_segments must be a positive integer (got '$NUM_SEGMENTS')." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/standby_train.sh"

if [ ! -f "$WORKER" ]; then
    echo "Error: worker script not found at $WORKER" >&2
    exit 1
fi

echo "Submitting $NUM_SEGMENTS segment(s) for run '$RUN_NAME'."
echo "Per-segment command: standby_train.sh $RUN_NAME $*"
echo

prev_job=""
for (( seg=1; seg<=NUM_SEGMENTS; seg++ )); do
    if [ -z "$prev_job" ]; then
        job_id="$(sbatch --parsable "$WORKER" "$RUN_NAME" "$@")"
        echo "Segment $seg: job $job_id (starts when scheduled)"
    else
        job_id="$(sbatch --parsable --dependency=afterany:"$prev_job" "$WORKER" "$RUN_NAME" "$@")"
        echo "Segment $seg: job $job_id (starts after job $prev_job)"
    fi
    prev_job="$job_id"
done

echo
echo "Chain submitted. Monitor with:  squeue -u \"$USER\" --name=wintertrain"
echo "Cancel the whole chain with:    scancel --name=wintertrain -u \"$USER\""
