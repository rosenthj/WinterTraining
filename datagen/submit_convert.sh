#!/bin/bash
#
# Convert a campaign's shard PGNs into per-shard datasets as a SLURM job array.
# Step 2 of the pipeline: generate shards -> CONVERT SHARDS -> merge datasets.
#
# Usage:
#   ./submit_convert.sh <dataset_name> [max_parallel]
#
#   dataset_name   The campaign name given to submit_datagen.sh; the shard PGNs
#                  $RAW_DIR/<dataset_name>_shard*.pgn are converted.
#   max_parallel   Max conversions running at once (array '%' throttle).
#                  Default $MAX_PARALLEL_DEFAULT.
#
# Why this is a separate submission rather than part of datagen_worker.sh: conversion has
# to be repeatable without replaying the games. Every change to the extractor or to the
# forfeit handling means reconverting, and coupling the two would make that cost a fresh
# self-play campaign. It also keeps each datagen shard's full 3:55 available for games.
#
# Reruns are safe: a shard whose dataset already exists is skipped, so resubmitting the
# whole array after a partial failure only redoes what is missing (CONVERT_FORCE=1 to
# reconvert everything).
#
# Examples:
#   ./submit_convert.sh winter_v1
#   ./submit_convert.sh winter_v1 4
#   CONVERT_ARGS="--drop-abnormal --tb-relabel-prob 0.5" ./submit_convert.sh winter_v1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=datagen_config.sh
source "$SCRIPT_DIR/datagen_config.sh"

if [ "$#" -lt 1 ]; then
    sed -n '2,26p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'
    exit 1
fi

DATASET="$1"
MAX_PARALLEL="${2:-$MAX_PARALLEL_DEFAULT}"

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL" -lt 1 ]; then
    echo "Error: max_parallel must be a positive integer (got '$MAX_PARALLEL')." >&2
    exit 1
fi

CONVERTER="$SCRIPT_DIR/../src/pgn_to_dataset.py"
if [ ! -f "$CONVERTER" ]; then
    echo "Error: converter not found at $CONVERTER" >&2
    exit 1
fi
if [ -z "$TB_PATH" ]; then
    echo "Error: no tablebase path. Copy local_env.sh.example to local_env.sh in the" >&2
    echo "       repository root and set WINTER_TB_PATH, or export TB_PATH." >&2
    exit 1
fi
if [ ! -d "$TB_PATH" ]; then
    echo "Warning: tablebase directory not found on this node: $TB_PATH" >&2
    echo "         (fine if it only exists on the compute nodes)" >&2
fi

shopt -s nullglob
SHARDS=( "$RAW_DIR/${DATASET}_shard"*.pgn )
shopt -u nullglob
if [ "${#SHARDS[@]}" -eq 0 ]; then
    echo "Error: no shard PGNs matched $RAW_DIR/${DATASET}_shard*.pgn" >&2
    exit 1
fi

mkdir -p "$SHARD_DATASET_DIR" "$LOG_DIR"

# The manifest decouples array task ids from shard ids, which need not be contiguous once
# submit_datagen.sh has been used with a shard_offset to extend a campaign.
MANIFEST="$SHARD_DATASET_DIR/${DATASET}.manifest"
: > "$MANIFEST"
PENDING=0
for pgn in "${SHARDS[@]}"; do
    name="$(basename "$pgn" .pgn)"
    printf '%s\n' "$name" >> "$MANIFEST"
    if [ ! -f "$SHARD_DATASET_DIR/features_${name}.npz" ]; then
        PENDING=$(( PENDING + 1 ))
    fi
done
LAST=$(( ${#SHARDS[@]} - 1 ))
ARRAY_SPEC="0-${LAST}%${MAX_PARALLEL}"

echo "Submitting Winter shard conversion:"
echo "   dataset        : $DATASET"
echo "   shards         : ${#SHARDS[@]} ($PENDING not yet converted)"
echo "   manifest       : $MANIFEST"
echo "   max parallel   : $MAX_PARALLEL"
echo "   time/shard     : $CONVERT_TIME"
echo "   resources      : $CONVERT_CPUS cpus, ${CONVERT_GPUS_PER_NODE} gpu (idle), $CONVERT_MEM"
echo "   account/qos    : $SLURM_ACCOUNT / $SLURM_QOS / $SLURM_PARTITION"
echo "   tablebase      : $TB_PATH"
echo "   extra args     : ${CONVERT_ARGS:-none}"
echo "   shard datasets : $SHARD_DATASET_DIR/features_${DATASET}_shard<NNNNN>.npz"
echo

JOB_ID="$(sbatch --parsable \
    --account="$SLURM_ACCOUNT" \
    --qos="$SLURM_QOS" \
    --partition="$SLURM_PARTITION" \
    --gpus-per-node="$CONVERT_GPUS_PER_NODE" \
    --cpus-per-task="$CONVERT_CPUS" \
    --mem="$CONVERT_MEM" \
    --time="$CONVERT_TIME" \
    --job-name="$CONVERT_JOB_NAME" \
    --array="$ARRAY_SPEC" \
    --output="$LOG_DIR/%x_%A_%a.out" \
    --export=ALL,DATAGEN_SCRIPT_DIR="$SCRIPT_DIR",MANIFEST="$MANIFEST",SHARD_INDEX_BASE=0 \
    "$SCRIPT_DIR/convert_worker.sh")"

echo "Submitted array job $JOB_ID  (tasks $ARRAY_SPEC)"
echo
echo "Monitor:        squeue -u \"$USER\" --name=$CONVERT_JOB_NAME"
echo "Cancel:         scancel $JOB_ID"
echo "Logs:           $LOG_DIR/${CONVERT_JOB_NAME}_${JOB_ID}_*.out"
echo
echo "When done, merge the shard datasets into training datasets (step 3):"
echo "   cd $SCRIPT_DIR/../src"
echo "   python merge_datasets.py --glob '$SHARD_DATASET_DIR/features_${DATASET}_shard*.npz' \\"
echo "                            --dry-run"
echo "   # then rerun with --start-version <N> to write them"
