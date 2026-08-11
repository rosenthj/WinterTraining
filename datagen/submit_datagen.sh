#!/bin/bash
#
# Launch a Winter DFRC self-play data-generation campaign as a throttled SLURM
# job array. Each array task is one ~4-hour shard (datagen_worker.sh) that plays
# from a disjoint slice of the DFRC opening book and writes one PGN shard.
#
# Usage:
#   ./submit_datagen.sh <dataset_name> <num_shards> [max_parallel] [shard_offset]
#
#   dataset_name   Name for this campaign; appears in shard filenames and the
#                  default merged-dataset prefix.
#   num_shards     How many 4h shard jobs to run.
#   max_parallel   Max shards running at once (array '%' throttle). Default 6.
#                  The cluster allows at most 24; use fewer to leave room for
#                  your primary research jobs.
#   shard_offset   First shard id (default 0). Use this to ADD more shards to an
#                  existing dataset later without reusing the same openings,
#                  e.g. if you already ran shards 0-6, start the next batch at 7.
#
# Examples:
#   # ~250k games (≈7 shards at ~35k games/shard), up to 6 running at once:
#   ./submit_datagen.sh winter_v1 7
#
#   # Be gentle on the cluster: only 2 shards at a time:
#   ./submit_datagen.sh winter_v1 7 2
#
#   # Add 7 more shards to winter_v1 later, with fresh openings:
#   ./submit_datagen.sh winter_v1 7 6 7
#
# After the array finishes, convert the shards and merge (steps 2 and 3):
#   ./submit_convert.sh winter_v1


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=datagen_config.sh
source "$SCRIPT_DIR/datagen_config.sh"

if [ "$#" -lt 2 ]; then
    sed -n '2,40p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'
    exit 1
fi

DATASET="$1"
NUM_SHARDS="$2"
MAX_PARALLEL="${3:-$MAX_PARALLEL_DEFAULT}"
SHARD_OFFSET="${4:-0}"

# --- validation --------------------------------------------------------------
for v in NUM_SHARDS MAX_PARALLEL SHARD_OFFSET; do
    if ! [[ "${!v}" =~ ^[0-9]+$ ]]; then
        echo "Error: $v must be a non-negative integer (got '${!v}')." >&2
        exit 1
    fi
done
if [ "$NUM_SHARDS" -lt 1 ]; then
    echo "Error: num_shards must be >= 1." >&2
    exit 1
fi
if [ "$MAX_PARALLEL" -lt 1 ] || [ "$MAX_PARALLEL" -gt 24 ]; then
    echo "Error: max_parallel must be between 1 and 24 (got $MAX_PARALLEL)." >&2
    exit 1
fi
if [ ! -x "$ENGINE" ]; then
    echo "Error: Winter binary not executable at $ENGINE" >&2
    exit 1
fi
if [ ! -f "$OPENINGS" ]; then
    echo "Error: openings file not found at $OPENINGS" >&2
    exit 1
fi
RUNNER="$FASTCHESS"
if ! command -v "$RUNNER" >/dev/null 2>&1 && [ ! -x "$RUNNER" ]; then
    echo "Error: fastchess not found / not executable: $RUNNER" >&2
    exit 1
fi

mkdir -p "$RAW_DIR" "$LOG_DIR"

LAST=$(( SHARD_OFFSET + NUM_SHARDS - 1 ))
ARRAY_SPEC="${SHARD_OFFSET}-${LAST}%${MAX_PARALLEL}"

echo "Submitting Winter DFRC datagen campaign:"
echo "   dataset        : $DATASET"
echo "   runner         : $RUNNER"
echo "   shards         : $NUM_SHARDS (ids $SHARD_OFFSET..$LAST)"
echo "   max parallel   : $MAX_PARALLEL"
echo "   time/shard     : $SLURM_TIME"
echo "   resources      : $SLURM_CPUS cpus, ${SLURM_GPUS_PER_NODE} gpu, $SLURM_MEM"
echo "   account/qos    : $SLURM_ACCOUNT / $SLURM_QOS / $SLURM_PARTITION"
echo "   shard pgns     : $RAW_DIR/${DATASET}_shard<NNNNN>.pgn"
echo

JOB_ID="$(sbatch --parsable \
    --account="$SLURM_ACCOUNT" \
    --qos="$SLURM_QOS" \
    --partition="$SLURM_PARTITION" \
    --gpus-per-node="$SLURM_GPUS_PER_NODE" \
    --cpus-per-task="$SLURM_CPUS" \
    --mem="$SLURM_MEM" \
    --time="$SLURM_TIME" \
    --job-name="$SLURM_JOB_NAME" \
    --array="$ARRAY_SPEC" \
    --output="$LOG_DIR/%x_%A_%a.out" \
    --export=ALL,DATAGEN_SCRIPT_DIR="$SCRIPT_DIR",DATASET="$DATASET" \
    "$SCRIPT_DIR/datagen_worker.sh")"

echo "Submitted array job $JOB_ID  (tasks $ARRAY_SPEC)"
echo
echo "Monitor:        squeue -u \"$USER\" --name=$SLURM_JOB_NAME"
echo "Cancel:         scancel $JOB_ID"
echo "Logs:           $LOG_DIR/${SLURM_JOB_NAME}_${JOB_ID}_*.out"
echo
echo "When done, convert the shards to per-shard datasets (step 2):"
echo "   $SCRIPT_DIR/submit_convert.sh $DATASET"
