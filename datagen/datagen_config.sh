#!/bin/bash
#
# Shared configuration for Winter DFRC self-play data generation.
# Sourced by datagen_worker.sh and submit_datagen.sh. Override any value by
# exporting it in your environment before calling those scripts, e.g.:
#
#     CONCURRENCY=16 ./submit_datagen.sh run1 7
#
# All paths are resolved relative to this file's directory unless absolute.

# Directory containing this config (and the scripts / Winter binary / book).
DATAGEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Machine-specific paths come from the git-ignored ../local_env.sh so that no filesystem
# location is committed; see local_env.sh.example. Sourced first, before anything below
# takes a default, so local_env.sh can override any of them. Variables already exported in
# the environment still win, since every setting below uses ${VAR:-default}.
if [ -f "$DATAGEN_DIR/../local_env.sh" ]; then
    # shellcheck source=/dev/null
    source "$DATAGEN_DIR/../local_env.sh"
fi

# ---- Engine -----------------------------------------------------------------
# The engine must be built on the cluster, so its location varies; set WINTER_ENGINE in
# local_env.sh if it is not at datagen/Winter.
ENGINE="${ENGINE:-${WINTER_ENGINE:-$DATAGEN_DIR/Winter}}"   # Winter UCI binary (CPU)
HASH_MB="${HASH_MB:-64}"                        # transposition table per engine
THREADS="${THREADS:-1}"                         # 1 search thread per engine

# ---- Match runner -----------------------------------------------------------
# fastchess drives the games (static binary, no compile needed). Its [Termination]
# tags and -pgnout options are what the dataset converter's game_filter relies on,
# so the runner is not interchangeable: cutechess support was removed rather than
# left as an unexercised path that could silently diverge on exactly those tags.
FASTCHESS="${FASTCHESS:-$DATAGEN_DIR/fastchess}"
# If a runner is provided as a module, set this to `module load` it, e.g.:
# RUNNER_MODULE="${RUNNER_MODULE:-fastchess}"

# ---- Openings (DFRC) --------------------------------------------------------
OPENINGS="${OPENINGS:-$DATAGEN_DIR/DFRC.epd}"   # one DFRC position per line (EPD)
VARIANT="${VARIANT:-fischerandom}"              # required for (D)FRC castling

# Opening selection order:
#   random     - each shard shuffles the whole book with its own seed and plays
#                through it. Different seed => different sample, so every shard
#                (and every run) draws a different random subset. This is the
#                default. The seed is derived from the SLURM array job id (see
#                SEED_OVERRIDE below), so each `sbatch` automatically gets a
#                fresh sample with no manual bookkeeping.
#   sequential - each shard reads a disjoint contiguous slice of the book
#                (start = SHARD_ID * OPENINGS_PER_SHARD). Fully reproducible and
#                guaranteed non-overlapping, but identical openings across runs
#                with the same shard ids.
ORDER="${ORDER:-random}"

# Only used when ORDER=sequential: size of each shard's contiguous slice. With
# one game per opening a 4h shard consumes ~40k openings, so keep this safely
# above that to avoid a shard exhausting its slice before the time limit.
OPENINGS_PER_SHARD="${OPENINGS_PER_SHARD:-60000}"

# Seed control. By default each shard's RNG seed is derived from the SLURM array
# job id + shard id, so every submission samples different openings (with
# ORDER=random) while parallel shards in one submission stay distinct. Set
# SEED_OVERRIDE to a fixed integer to reproduce an exact previous sample
# (shard i then uses SEED_OVERRIDE + i).
SEED_OVERRIDE="${SEED_OVERRIDE:-}"

# ---- Match / time control ---------------------------------------------------
# Games are played on the clock (not to a fixed node count): a node limit would
# make each (opening, engine) pair deterministic, so any opening drawn twice
# would yield a byte-identical game to deduplicate.
TC="${TC:-3+0.03}"                              # 3 s base + 0.03 s increment

# Grace period (milliseconds) before an engine that overshoots its clock is
# forfeited. fastchess defaults to 0, i.e. a single millisecond of scheduler
# jitter loses the game -- and with a 0.03 s increment, a game deep in an endgame
# is living entirely inside that jitter. A forfeit hands the win to the *losing*
# side, which is how a lone king ends up "winning" a game. On a shared,
# non-pinned cluster node a margin is a necessity, not a nicety.
#
# Note what the margin does and does not do. It forgives the *flag*, not the
# *time*: the overage is still debited in full, so an engine that overshoots by
# 90 ms is left around -60 ms once the 0.03 s increment lands, begins its next
# move already over budget, and forfeits on that move instead. The margin buys
# one move of grace, not a reset -- which is why CONCURRENCY headroom below
# matters as much as this setting does.
# Set to 0 if you are reusing this config to measure Elo rather than make data.
TIMEMARGIN="${TIMEMARGIN:-100}"

# Games in parallel. Each game has one engine thinking at a time, so this is
# roughly one core per game, but leaving no headroom is what produced forfeit
# rates between 4% and 52% across desk_v303..v317 -- and because the margin above
# cannot give time back, a single scheduling stall is unrecoverable. Kept
# comfortably under SLURM_CPUS rather than up against it.
CONCURRENCY="${CONCURRENCY:-24}"

# Adjudication (matches the settings used for previous local datagen runs).
DRAW_ADJ="${DRAW_ADJ:-movenumber=50 movecount=10 score=5}"
# Resign adjudication is intentionally OFF (empty). To enable, e.g.:
#   RESIGN_ADJ="movecount=5 score=1000 twosided=true"
RESIGN_ADJ="${RESIGN_ADJ:-}"

# ---- Output -----------------------------------------------------------------
# Write engine eval/depth/time as move comments ({+0.34/13 0.11s}). Set to
# false for minimal PGNs (moves + result + start FEN only), which are much
# smaller -- use this if your PGN->(position, result) converter ignores evals.
#
# WARNING: PGN_COMMENTS=false also strips the [Termination] tag, the most reliable
# record of *why* a game ended, and disables timeleft tracking. Without the tag a
# time forfeit has to be inferred from the final position by src/game_filter.py --
# exact only while RESIGN_ADJ is empty, and blind to a forfeit that lands on a
# mate. Keep this true; the shards stay on scratch, so the comments cost nothing.
PGN_COMMENTS="${PGN_COMMENTS:-true}"

RAW_DIR="${RAW_DIR:-$DATAGEN_DIR/data/raw}"         # per-shard PGN shards
LOG_DIR="${LOG_DIR:-$DATAGEN_DIR/logs}"

# ---- Shard conversion (step 2) ----------------------------------------------
# Per-shard datasets from submit_convert.sh, before merge_datasets.py combines them.
# Deliberately NOT ../datasets/: loader.discover_dataset_tags globs that directory for
# features_desk_v*.npz, and a shard sitting there would be picked up as a dataset in its
# own right.
SHARD_DATASET_DIR="${SHARD_DATASET_DIR:-$DATAGEN_DIR/data/shards}"

# Syzygy WDL tablebases. Only WDL (.rtbw) is read; DTZ is never probed.
TB_PATH="${TB_PATH:-${WINTER_TB_PATH:-}}"

# Environment for the Python converter, which lives in ../src of this repository.
CONVERT_CONDA_ENV="${CONVERT_CONDA_ENV:-${WINTER_CONDA_ENV:-}}"

# Extra pgn_to_dataset.py flags for every shard. --drop-abnormal repairs or drops games
# whose result was decided by the clock rather than by chess; harmless on clean shards.
CONVERT_ARGS="${CONVERT_ARGS:---drop-abnormal}"

# Conversion is CPU-only and single-threaded, and a ~40k-game shard is minutes of work
# rather than hours, so it asks for far less than a datagen shard -- except for the GPU.
# A GPU-less job cannot be submitted to these partitions, and the A100 partitions have far
# more nodes than the others, so asking for one A100 schedules sooner than asking for a
# lesser GPU. It sits idle for the whole conversion either way.
CONVERT_GPUS_PER_NODE="${CONVERT_GPUS_PER_NODE:-1}"
CONVERT_CPUS="${CONVERT_CPUS:-2}"
CONVERT_MEM="${CONVERT_MEM:-8g}"
CONVERT_TIME="${CONVERT_TIME:-1:30:00}"
CONVERT_JOB_NAME="${CONVERT_JOB_NAME:-winterconv}"

# ---- SLURM (datagen jobs) ---------------------------------------------------
# Winter is a CPU engine and does not use the GPU, but this cluster only has
# GPU partitions, so we request 1 A100 to satisfy the partition.
SLURM_ACCOUNT="${SLURM_ACCOUNT:-antoniob}"
SLURM_QOS="${SLURM_QOS:-standby}"
SLURM_PARTITION="${SLURM_PARTITION:-a100-80gb}"
SLURM_CPUS="${SLURM_CPUS:-32}"
SLURM_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-1}"
SLURM_MEM="${SLURM_MEM:-16g}"
SLURM_TIME="${SLURM_TIME:-3:55:00}"                # wall-clock per job (< 4h)
SLURM_JOB_NAME="${SLURM_JOB_NAME:-winterdata}"

# Stop the match runner gracefully this many seconds into the job, leaving
# margin for it to finish in-flight games and flush the PGN before SLURM's hard
# kill. 3:50:00 = 13800 s, i.e. ~5 min of margin before the 3:55:00 wall limit.
MATCH_TIMEOUT="${MATCH_TIMEOUT:-13800}"

# Default parallelism cap for the job array (cluster allows at most 24).
MAX_PARALLEL_DEFAULT="${MAX_PARALLEL_DEFAULT:-6}"
