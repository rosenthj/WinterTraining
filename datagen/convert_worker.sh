#!/bin/bash
#
# Convert ONE shard PGN into a per-shard dataset. Step 2 of the pipeline; one array task
# per shard, submitted by submit_convert.sh.
#
# The shard to convert is read from a manifest file (one shard base name per line) using
# SLURM_ARRAY_TASK_ID as a 0-based line index, rather than reconstructing the name from the
# task id. Shard ids are not necessarily contiguous -- submit_datagen.sh takes a
# shard_offset so later batches can be appended to a campaign -- and a manifest also lets a
# rerun cover only the shards that actually failed.
#
# Expects MANIFEST and SHARD_INDEX_BASE in the environment (set by submit_convert.sh).

set -euo pipefail

# SLURM runs a *copy* of this script from /var/spool/slurm/job*/slurm_script, so
# ${BASH_SOURCE[0]} points into the spool directory and the config is not beside it. The
# submitter knows the real location and exports it; SLURM_SUBMIT_DIR is a fallback for a
# hand-submitted job, and BASH_SOURCE for running the script directly.
SCRIPT_DIR="${DATAGEN_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [ ! -f "$SCRIPT_DIR/datagen_config.sh" ]; then
    echo "Error: datagen_config.sh not found in $SCRIPT_DIR" >&2
    echo "       Submit via submit_convert.sh, which exports DATAGEN_SCRIPT_DIR." >&2
    exit 1
fi
# shellcheck source=datagen_config.sh
source "$SCRIPT_DIR/datagen_config.sh"

MANIFEST="${MANIFEST:?convert_worker.sh: MANIFEST not set (submit via submit_convert.sh)}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(( TASK_ID - ${SHARD_INDEX_BASE:-0} + 1 ))

NAME="$(sed -n "${LINE}p" "$MANIFEST")"
if [ -z "$NAME" ]; then
    echo "Error: manifest $MANIFEST has no entry at line $LINE (task $TASK_ID)" >&2
    exit 1
fi

FEATURES="$SHARD_DATASET_DIR/features_${NAME}.npz"
TARGETS="$SHARD_DATASET_DIR/targets_${NAME}.npz"

echo "=================================================================="
echo " Winter shard conversion"
echo "   shard             : $NAME"
echo "   task id           : $TASK_ID (manifest line $LINE)"
echo "   host              : $(hostname)"
echo "   input pgn         : $RAW_DIR/${NAME}.pgn"
echo "   output            : $FEATURES"
echo "   tablebase         : $TB_PATH"
echo "   extra args        : ${CONVERT_ARGS:-none}"
echo "=================================================================="

if [ ! -f "$RAW_DIR/${NAME}.pgn" ]; then
    echo "Error: shard PGN not found: $RAW_DIR/${NAME}.pgn" >&2
    exit 1
fi

# Idempotent: a resubmitted array skips shards that already converted, so a partially
# failed run can simply be resubmitted whole. Set CONVERT_FORCE=1 to reconvert.
if [ -f "$FEATURES" ] && [ -f "$TARGETS" ] && [ "${CONVERT_FORCE:-0}" != "1" ]; then
    echo "Already converted, skipping (CONVERT_FORCE=1 to redo)."
    exit 0
fi

# Only manage the environment when there is one to manage. Guarding this lets the worker be
# run by hand off-cluster (or under an already-activated env) to debug a single shard.
if [ "${CONVERT_SKIP_ENV:-0}" != "1" ] && command -v module >/dev/null 2>&1; then
    : "${CONVERT_CONDA_ENV:?set WINTER_CONDA_ENV in local_env.sh (see local_env.sh.example)}"
    module load conda
    conda activate "$CONVERT_CONDA_ENV"
fi

mkdir -p "$SHARD_DATASET_DIR"

# pgn_to_dataset.py builds its paths by string concatenation, so the directories must
# carry a trailing slash.
cd "$SCRIPT_DIR/../src"

ARGS=(--name "$NAME"
      --pgn-dir "${RAW_DIR%/}/"
      --out-dir "${SHARD_DATASET_DIR%/}/"
      --tablebase "$TB_PATH")
# shellcheck disable=SC2206  # deliberate word splitting: CONVERT_ARGS is a flag string
[ -n "${CONVERT_ARGS:-}" ] && ARGS+=( ${CONVERT_ARGS} )

echo "Running: python -u pgn_to_dataset.py ${ARGS[*]}"
python -u pgn_to_dataset.py "${ARGS[@]}"

echo
echo "Wrote $FEATURES"
ls -l "$FEATURES" "$TARGETS"
