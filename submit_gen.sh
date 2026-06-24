#!/bin/bash
#
# Fan out PGN -> dataset conversion across the cluster: one independent SLURM job per
# PGN file. Unlike submit_chain.sh there is no dependency chain -- the files are
# independent, so the jobs run in parallel (subject to the queue) for maximum throughput.
#
# Usage:
#   ./submit_gen.sh <name> [<name> ...]
#
# Examples:
#   ./submit_gen.sh desk_v300 desk_v301 desk_v302
#   ./submit_gen.sh desk_v{300..302}          # brace expansion happens in your shell
#
# Each job runs:  generate_dataset.sh <name>
# which calls   pgn_to_dataset.py --name <name> --tablebase "$TB_PATH"
#
# Override the tablebase path for every job:  TB_PATH=/some/path ./submit_gen.sh ...

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <name> [<name> ...]" >&2
    echo "  e.g. $0 desk_v300 desk_v301 desk_v302" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/generate_dataset.sh"

if [ ! -f "$WORKER" ]; then
    echo "Error: worker script not found at $WORKER" >&2
    exit 1
fi

# Forward TB_PATH to the jobs if it is set in this shell (otherwise the worker default wins).
EXPORT_ARG=()
if [ -n "${TB_PATH:-}" ]; then
    EXPORT_ARG=(--export=ALL,TB_PATH="$TB_PATH")
fi

echo "Submitting $# conversion job(s)."
for name in "$@"; do
    pgn="$SCRIPT_DIR/pgns/$name.pgn"
    if [ ! -f "$pgn" ]; then
        echo "  Warning: $pgn not found on the login node (continuing; it is read on the compute node)." >&2
    fi
    job_id="$(sbatch --parsable "${EXPORT_ARG[@]}" "$WORKER" "$name")"
    echo "  $name -> job $job_id"
done

echo
echo "Monitor with:  squeue -u \"$USER\" --name=wintergen"
echo "Cancel all:    scancel --name=wintergen -u \"$USER\""
echo "Logs:          logs/wintergen_<jobid>.out"
