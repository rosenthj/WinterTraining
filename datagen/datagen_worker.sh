#!/bin/bash
#SBATCH -A antoniob
#SBATCH -q standby
#SBATCH --partition=a100-80gb
#SBATCH --gpus-per-node=1
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:55:00
#SBATCH --job-name=winterdata
#SBATCH --output=logs/%x_%A_%a.out
#
# One ~4-hour Winter DFRC self-play data-generation shard.
#
# Normally launched as a SLURM job array by submit_datagen.sh, which sets:
#   DATASET                 - dataset name (used in the shard filename)
#   SLURM_ARRAY_TASK_ID     - the shard id (determines the opening slice + seed)
#
# Each shard plays Winter vs Winter from a disjoint slice of the DFRC book and
# writes finished games (with engine evals in the move comments) to one PGN:
#     data/raw/<DATASET>_shard<NNNNN>.pgn
#
# The #SBATCH directives above are fallback defaults; submit_datagen.sh passes
# explicit overrides (account, partition, cpus, time, array throttle, ...) on
# the sbatch command line, which take precedence.

set -euo pipefail

# Resolve and load shared config. SLURM runs a *copy* of this script from the spool
# directory, so ${BASH_SOURCE[0]} does not point at the checkout; the submitter exports the
# real location. SLURM_SUBMIT_DIR only happens to work when sbatch was run from this
# directory, so it is a fallback rather than the primary.
SCRIPT_DIR="${DATAGEN_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [ ! -f "$SCRIPT_DIR/datagen_config.sh" ]; then
    echo "Error: datagen_config.sh not found in $SCRIPT_DIR" >&2
    echo "       Submit via submit_datagen.sh, which exports DATAGEN_SCRIPT_DIR." >&2
    exit 1
fi
# shellcheck source=datagen_config.sh
source "$SCRIPT_DIR/datagen_config.sh"

# Optionally load a runner module on the cluster.
if [ -n "${RUNNER_MODULE:-}" ]; then
    module load "$RUNNER_MODULE"
fi

RUNNER="$FASTCHESS"

DATASET="${DATASET:-datagen}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "$RAW_DIR" "$LOG_DIR"

# Number of openings in the book (one position per line).
NUM_OPENINGS="$(wc -l < "$OPENINGS")"

# RNG seed for this shard. With ORDER=random the seed selects which openings are
# sampled, so it must differ per shard AND per run. Derive it from the SLURM
# array job id (unique per submission) mixed with the shard id, unless the user
# pinned SEED_OVERRIDE for reproducibility. Kept within 31 bits for the runners.
if [ -n "${SEED_OVERRIDE:-}" ]; then
    SEED=$(( (SEED_OVERRIDE + SHARD_ID) % 2147483647 ))
else
    SEED_BASE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%s)}}"
    SEED=$(( (SEED_BASE_ID * 2654435761 + SHARD_ID * 40503) % 2147483647 ))
fi

PGN="$RAW_DIR/$(printf '%s_shard%05d.pgn' "$DATASET" "$SHARD_ID")"

# Build adjudication args.
ADJ_ARGS=()
if [ -n "$DRAW_ADJ" ]; then
    # shellcheck disable=SC2206
    ADJ_ARGS+=( -draw $DRAW_ADJ )
fi
if [ -n "$RESIGN_ADJ" ]; then
    # shellcheck disable=SC2206
    ADJ_ARGS+=( -resign $RESIGN_ADJ )
fi

# Opening-selection args and the -rounds cap depend on ORDER.
#   random     - shuffle the whole book (seeded by -srand); cap rounds at the
#                book size (timeout stops the shard long before that).
#   sequential - read this shard's disjoint contiguous slice via start=.
case "$ORDER" in
    random)
        OPENING_ARGS=( -openings file="$OPENINGS" format=epd order=random )
        ROUNDS="$NUM_OPENINGS"
        OPENING_DESC="random shuffle, seed $SEED (whole book of $NUM_OPENINGS)"
        ;;
    sequential)
        START=$(( (SHARD_ID * OPENINGS_PER_SHARD) % NUM_OPENINGS + 1 ))
        OPENING_ARGS=( -openings file="$OPENINGS" format=epd order=sequential start="$START" )
        ROUNDS="$OPENINGS_PER_SHARD"
        OPENING_DESC="sequential slice start=$START (size $OPENINGS_PER_SHARD, wraps)"
        ;;
    *)
        echo "Error: ORDER must be 'random' or 'sequential' (got '$ORDER')" >&2
        exit 1 ;;
esac

echo "=================================================================="
echo " Winter DFRC datagen shard"
echo "   dataset           : $DATASET"
echo "   shard id          : $SHARD_ID"
echo "   host              : $(hostname)"
echo "   runner            : $RUNNER"
echo "   version           : $("$RUNNER" --version 2>/dev/null | head -1)"
echo "   engine            : $ENGINE"
echo "   openings          : $OPENINGS ($NUM_OPENINGS positions)"
echo "   opening order     : $OPENING_DESC"
echo "   time control      : $TC (timemargin ${TIMEMARGIN:-0} ms)"
echo "   concurrency       : $CONCURRENCY"
echo "   draw adjudication : ${DRAW_ADJ:-none}"
echo "   resign adjud.     : ${RESIGN_ADJ:-none}"
echo "   seed              : $SEED"
echo "   pgn comments      : $PGN_COMMENTS"
echo "   output pgn        : $PGN"
echo "   match timeout     : ${MATCH_TIMEOUT}s"
echo "=================================================================="

# Build the command.
#
# -games 1 plays each opening exactly once (no color-reversed replay -- that is
# a variance-reduction trick for Elo testing, not useful for datagen, where we
# want maximum opening diversity). fastchess writes a game to the PGN only once
# it has finished, and timeout -s INT lets it stop after in-flight games before
# SLURM's hard kill, so shards are never truncated mid-game.
CMD=( "$RUNNER"
    -engine name=Winter1 cmd="$ENGINE"
    -engine name=Winter2 cmd="$ENGINE" )

# timemargin tolerates scheduler jitter on a shared node instead of forfeiting
# the game (a forfeit awards the win to the side that was losing on the board).
EACH_ARGS=( proto=uci tc="$TC" )
[ -n "${TIMEMARGIN:-}" ] && EACH_ARGS+=( timemargin="$TIMEMARGIN" )

CMD+=( -each "${EACH_ARGS[@]}"
    option.Hash="$HASH_MB" option.Threads="$THREADS" option.UCI_Chess960=true )

CMD+=( -variant "$VARIANT"
    "${OPENING_ARGS[@]}"
    -games 1 -rounds "$ROUNDS"
    "${ADJ_ARGS[@]}"
    -concurrency "$CONCURRENCY"
    -recover
    -srand "$SEED" )

# Minimal PGN (no eval/depth/time comments) when PGN_COMMENTS=false -- smaller
# files for converters that only need moves + result + start position.
case "$PGN_COMMENTS" in
    true|false) ;;
    *) echo "Error: PGN_COMMENTS must be 'true' or 'false' (got '$PGN_COMMENTS')" >&2
       exit 1 ;;
esac

# fastchess caps concurrency at hardware threads unless forced; we sized the
# SLURM allocation deliberately, so force our requested concurrency.
# Disable periodic autosave (we never resume) and point the interrupt-time
# resume file at a per-shard path, so parallel array tasks don't clobber a
# shared config.json in the submit dir.
#
# timeleft=true records the clock remaining after each move. It costs a few bytes
# per move and is the only early warning available: clocks trending toward zero
# show up before any game is actually forfeited.
CONFIG_JSON="$RAW_DIR/$(printf '%s_shard%05d.config.json' "$DATASET" "$SHARD_ID")"
PGN_MIN=false; [ "$PGN_COMMENTS" = "false" ] && PGN_MIN=true
CMD+=( -force-concurrency
    -autosaveinterval 0 -config outname="$CONFIG_JSON"
    -pgnout file="$PGN" notation=san min="$PGN_MIN" timeleft="$PGN_COMMENTS" )

set +e
timeout -s INT "${MATCH_TIMEOUT}" "${CMD[@]}"
rc=$?
set -e

# timeout exits 124 when it had to stop the runner at the wall margin: expected.
if [ "$rc" -eq 124 ]; then
    echo "Runner stopped at the time limit (expected); partial shard saved."
    rc=0
fi

# Remove the (unused) fastchess resume file; we never resume shards.
[ -n "${CONFIG_JSON:-}" ] && rm -f "$CONFIG_JSON"

GAMES=0
[ -f "$PGN" ] && GAMES="$(grep -c '^\[Event ' "$PGN" || true)"
echo "Shard $SHARD_ID finished: $GAMES games in $PGN (runner rc=$rc)"

# Report how the games ended. Anything other than 'normal' / 'adjudication' was
# decided by something outside the game -- overwhelmingly time forfeits, which
# award the win to the side that was losing on the board and so poison the
# labels. src/game_filter.py repairs or removes them at conversion, but the rate is
# worth watching here: a climbing forfeit rate means the node is too loaded for
# this time control.
if [ -f "$PGN" ] && [ "$GAMES" -gt 0 ]; then
    echo "Termination breakdown:"
    grep -h '^\[Termination ' "$PGN" | sort | uniq -c | sort -rn | sed 's/^/  /'
    BAD="$(grep -c '^\[Termination "\(time forfeit\|illegal move\|stalled connection\|abandoned\)"\]' "$PGN" || true)"
    if [ "$BAD" -gt 0 ]; then
        PCT="$(awk -v b="$BAD" -v g="$GAMES" 'BEGIN{printf "%.3f", 100*b/g}')"
        echo "  -> $BAD/$GAMES games ($PCT%) ended abnormally; conversion will repair or drop them"
    fi
fi
exit "$rc"
