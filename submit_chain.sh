#!/bin/bash
#
# Enqueue a training run as a job array of ~4-hour SLURM segments, throttled to one task at
# a time (--array=1-N%1), so a long run survives the standby queue's 4h-per-job limit while
# occupying a single line in squeue instead of N.
#
# Usage:
#   ./submit_chain.sh <num_segments> <run_name> [train_net.py args...]
#
# Examples:
#   ./submit_chain.sh 6 baseline --datasets all --exclude vEnd --batch-size 256
#   ./submit_chain.sh 3 frc_v2  --datasets 200-221 --init-lr 0.008 --min-lr 0.0001
#
# Each task runs:  standby_train.sh <run_name> [train args...]
# which auto-resumes from the newest checkpoint in ../models/<run_name>/.
#
# Why an array rather than a chain of --dependency=afterany jobs: %1 runs the segments
# strictly one at a time just as the dependencies did, but the run is now one job id.
# squeue shows it as one line, `scancel <id>` stops the whole run, and `scancel <id>_[3-12]`
# drops the queued tail while letting the running segment finish -- none of which the chain
# could do without collecting every segment's id. Every task is the same command, so unlike
# the datagen array there is no manifest mapping a task index to its work.
#
# Segment *order* is not something SLURM guarantees, only that one runs at a time, and
# nothing here depends on it: --auto-resume loads the newest checkpoint, so whichever task
# runs next continues from wherever the run actually got to.
#
# Failure behaviour is unchanged. A task that hits the 4h limit, is preempted, or dies exits
# non-zero and the array carries on with the next task, which is what makes the segmenting
# work at all; the price, as before, is that a genuinely broken run also keeps marching, so
# watch the first segment's log under logs/.
#
# The job name is the run name, so concurrent runs are told apart at a glance in squeue
# (the default squeue format truncates the name field to 8 characters).

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

echo "Submitting $NUM_SEGMENTS segment(s) for run '$RUN_NAME', one at a time."
echo "Per-segment command: standby_train.sh $RUN_NAME $*"
echo

# --job-name and --output override the worker's own #SBATCH directives, which stay as they
# are so it can still be sbatch'd standalone. %A_%a names the logs by the array job id and
# segment number; %j would name them by each task's individual job id, which is not the id
# the run is monitored and cancelled under.
JOB_ID="$(sbatch --parsable \
    --array="1-${NUM_SEGMENTS}%1" \
    --job-name="$RUN_NAME" \
    --output="logs/%x_%A_%a.out" \
    "$WORKER" "$RUN_NAME" "$@")"

echo "Array job $JOB_ID: $NUM_SEGMENTS segment(s), one running at a time."
echo
echo "Monitor this run:      squeue -u \"$USER\" --name=\"$RUN_NAME\""
echo "All your jobs:         squeue -u \"$USER\""
echo "Logs:                  logs/${RUN_NAME}_${JOB_ID}_<segment>.out"
echo "Cancel the whole run:  scancel $JOB_ID"
if [ "$NUM_SEGMENTS" -gt 1 ]; then
    echo "Drop the queued tail:  scancel ${JOB_ID}_[2-${NUM_SEGMENTS}]"
fi
