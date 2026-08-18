# Winter DFRC Data Generation (SLURM)

Self-play data generation for the Winter chess engine using DFRC (Double Fischer
Random) openings, driven by **fastchess** on a SLURM cluster.

## What this produces

Winter plays itself (`Winter` vs `Winter`) at **3+0.03** from Double Fischer
Random starting positions (`DFRC.epd`, 921,600 positions). By default
each game is written to a PGN with the engine's evaluation and remaining clock in
every move comment (`{+0.34/13 0.11s, tl=0.315s}` = score / depth / time / time
left). Shards are converted to `.npz` datasets and merged into the evenly sized
files that training loads; the PGNs stay on scratch.

## Match runner: fastchess

The games are run by **fastchess** (the static binary in this directory — already
on the server, nothing to compile).

```bash
./submit_datagen.sh winter_v1 7
```

fastchess is not interchangeable here. Its `[Termination]` tags are what
`src/game_filter.py` relies on to identify time forfeits, and `-pgnout timeleft=`
is what records the per-move clock. Earlier revisions also supported
`cutechess-cli` behind a `BACKEND` switch; that path was removed rather than left
unexercised, since a runner whose termination vocabulary silently differed would
break the filtering downstream without any visible error. Set the binary path
with `FASTCHESS=` (or `RUNNER_MODULE=` to `module load` it) in
`datagen_config.sh`.

Minor PGN notes: castling rights use X-FEN file letters (e.g. `HFhf`) and the tag
is `[Variant "Chess960"]` — both standard and parseable.

## Pipeline

Three stages, each a separate submission. Games are generated once; everything
downstream can be redone without replaying them.

```bash
# 1. Generate shard PGNs (~40k games each, one ~4h SLURM task per shard)
./submit_datagen.sh winter_v1 7

# 2. Convert each shard to a per-shard dataset (one short SLURM task per shard)
./submit_convert.sh winter_v1

# 3. Merge the shard datasets into evenly sized training datasets
cd ../src
python merge_datasets.py --glob '../datagen/data/shards/features_winter_v1_shard*.npz' \
                         --dry-run                    # check the split first
python merge_datasets.py --glob '../datagen/data/shards/features_winter_v1_shard*.npz' \
                         --start-version 318
```

Conversion is deliberately its own submission rather than a tail on the datagen
worker. Every change to the extractor or to how forfeits are handled means
reconverting, and coupling the two would make that cost a fresh self-play
campaign; it also leaves each datagen shard its full 3:55 for games. Reruns are
safe — a shard whose dataset already exists is skipped, so resubmitting after a
partial failure only redoes what is missing (`CONVERT_FORCE=1` to redo all).

### Building a validation set instead

The same three stages produce a validation set; only step 2 changes. Pass
`--positions-per-game 1` so each game contributes exactly one position, drawn
uniformly from the positions it yielded, and `--seed` so the set can be
regenerated identically later:

```bash
# 2'. Convert, keeping one position per game
CONVERT_ARGS="--drop-abnormal --positions-per-game 1 --seed 0" ./submit_convert.sh new_validation

# 3'. Merge into a single dataset, under a name the loader will not pick up as
#     training data (it globs features_desk_v*.npz)
python merge_datasets.py --glob '../datagen/data/shards/features_new_validation_shard*.npz' \
                         --base-name new_validation --start-version 1 --num-files 1
```

Then train against it with `--val-name new_validation1` (`merge_datasets.py`
always appends a version number, hence the `1`).

Why one per game: positions from the same game share a result, so a validation
set drawn from a few games has a far smaller effective sample size than its row
count suggests. It also removes a length bias — weighting every position equally
over-represents long games, which are not drawn as often as short ones. Measured
over 400 games from `winter_v1_shard00029`, the all-positions draw rate is 46.4%
against 51.0% one-per-game.

The subsampling happens *after* label extraction, not before: `data.py` threads
the result backwards through the game, so the Syzygy probe at the entry into the
endgame overwrites the recorded result for every earlier position. Drawing a
position first and labelling it on its own would disagree with how training
labels that same position.

### Why shards, not one big PGN

A 475k-game PGN takes roughly 1.8 hours of local-equivalent conversion, against
a 3:55 wall limit — a margin thin enough that a slower compute node alone
exhausts it, which is what made whole-file conversions time out. A ~40k-game
shard is about 9 minutes of the same work, so even a 4x slowdown leaves the
window largely unused, and the tasks run in parallel.

### Keep the eval comments

Each move carries an eval comment (`{+0.34/13 0.11s, tl=0.315s}`), often bigger
than the move itself. That costs nothing: the shards stay on scratch, and what
is downloaded now is the merged *datasets*, which are smaller than the PGNs they
came from (445k games is ~437 MB of PGN against ~316 MB of features + targets).

**Do not generate with `PGN_COMMENTS=false`.** It strips the `[Termination]`
tag, which is the most reliable record of *why* a game ended — without it a time
forfeit has to be inferred from the final position instead (see below), and
`timeleft` tracking goes with it.

## Mislabelled results: time forfeits

A game that ends on the clock still gets a decisive `[Result]`, and that result
is the opposite of the truth: the side that was *winning on the board* is the
side recorded as having lost. The signature is unmistakable in the data — games
where the side left with a lone king "won".

At 3+0.03 a long endgame is played entirely out of the 0.03 s increment, and
fastchess defaults to a **timemargin of 0**, so a single millisecond of
scheduler jitter forfeits the game. On a shared cluster node that is a real
risk. Three defences are in place:

1. **`TIMEMARGIN` (default 100 ms)** in `datagen_config.sh` gives the engine a
   grace period before it is forfeited, which suppresses most of them at the
   source. Set `TIMEMARGIN=0` if you ever reuse this config to measure Elo.

   The margin forgives the *flag*, not the *time*. Measured against fastchess
   1.8.0, the clock is `tl_after = tl_before + increment − time_spent` exactly,
   with the overage debited in full: an engine that overshoots by 90 ms is left
   around −60 ms once the increment lands, starts its next move already over
   budget, and forfeits on that move instead. One move of grace, not a reset.
2. **`CONCURRENCY` (default 24)**, kept below `SLURM_CPUS`. Because the margin
   cannot give time back, headroom is what actually prevents the failure; running
   up against the core count is what produced forfeit rates between 4% and 52%
   across `desk_v303`..`v317`.
3. **`src/game_filter.py`** in the converter removes or repairs the ones that
   still get through, before they reach a dataset.

`-pgnout timeleft=true` is enabled whenever `PGN_COMMENTS=true`, recording the
clock remaining after each move (`tl=0.315s`). That is the only early warning
available: clocks trending toward zero are visible before any game is forfeited.

Each shard also prints a termination breakdown at the end of its SLURM log, so a
rising forfeit rate is visible immediately:

```
Termination breakdown:
     39812 [Termination "normal"]
       147 [Termination "adjudication"]
        21 [Termination "time forfeit"]
  -> 21/39980 games (0.053%) ended abnormally and will be repaired or dropped at conversion
```

## Filtering: `src/game_filter.py`

Filtering happens inside the converter, so each shard is read once. Earlier
revisions used a separate `filter_pgn.py` pass over the PGNs; its checks live in
`game_filter.check_game` now.

| Check | Effect |
|---|---|
| `unfinished`     | `[Result "*"]` — **dropped**. Also a crash guard: `string_to_result_class` raises on it, which would abort the whole conversion rather than skip one game. |
| `unparsable`     | Movetext python-chess could not fully parse — **dropped**. Skipped moves mean the mainline is no longer the game that was played. |
| `terminal_state` | Final position is mate/stalemate/insufficient material but `[Result]` disagrees — **corrected from the position**, which is ground truth once the moves replay cleanly. |
| `termination`    | `[Termination]` other than `normal` / `adjudication` — treated as a forfeit (below). |
| positional       | A decisive result whose final position is not mate, stalemate, insufficient material, a repetition or a 50-move draw — treated as a forfeit. |

The positional test is the fallback for shards with no `[Termination]` tag, and
it is exact rather than heuristic **only while `RESIGN_ADJ` is empty**: with no
resign adjudication configured there is no way to end a game decisively except
by mate. An explicit `[Termination "adjudication"]` short-circuits it, so
enabling `RESIGN_ADJ` later will not silently discard legitimate adjudicated
wins.

`terminal_state` deserves note: it catches a forfeit the positional test
structurally cannot see. When a side flags on the very move that delivers mate,
the final position is a genuine checkmate and only the result is wrong. Left
alone those are the worst labels in the corpus — a full-length, normally played
game teaching that a mated king won. Rates run 0.10% (`desk_v312`) to 0.38%
(`desk_v315`), and every instance found so far is White mating and scored `0-1`.

### Salvaging forfeits with Syzygy

Dropping a forfeited game throws away a full game of positions, and most
forfeits happen in endgames — that is the mechanism, an engine living on the
increment while converting — which is exactly where Syzygy knows the truth. So
`--drop-abnormal` repairs where it can and drops only where it cannot:

- **Reached a tablebase position** → kept. The recorded result is only ever the
  seed of the backwards labelling walk, so the probe at the entry into the
  endgame overwrites it and every earlier position is labelled from the Syzygy
  verdict. Measured across `desk_v312` and `desk_v315`, the pre-endgame label of
  such a game matched the entry verdict in 324 of 324 cases, having overridden
  the recorded result in 203 of them.
- **Never reached one** → dropped. Nothing can override the result.

Salvage rate depends on how often forfeits reach an endgame at all: about 56% of
them on `desk_v312`, but only 16% on `desk_v315`. Cursed wins (WDL ±1 — won, but
drawn by the fifty-move rule) count as draws, matching the rules the games were
played under. Only WDL (`.rtbw`) files are read; DTZ is never probed.

## Files

| File | Purpose |
|------|---------|
| `datagen_config.sh`   | All tunables (engine, runner, TC, SLURM, openings, output). Sourced by the others. |
| `datagen_worker.sh`   | One ~4h shard (SLURM job). Runs fastchess on a disjoint opening slice. |
| `submit_datagen.sh`   | Launches a campaign as a throttled SLURM job array. |
| `submit_convert.sh`   | Converts a campaign's shard PGNs into per-shard datasets (step 2). |
| `convert_worker.sh`   | Converts one shard (array task of the above). |
| `DFRC.epd`            | The DFRC opening book (one position per line). |
| `Winter`              | The engine binary (UCI, CPU). |
| `fastchess`           | Default match runner (static binary). |

## Quick start

```bash
# 1. (one-time) confirm the fastchess binary runs on a compute node
#    (./fastchess --version). Override its path with FASTCHESS= if needed.

# 2. Launch ~250k games (≈7 shards at ~40k games/shard), <=6 shards at a time:
./submit_datagen.sh winter_v1 7

# 3. Watch it:
squeue -u "$USER" --name=winterdata

# 4. When the array finishes, convert the shards (step 2):
./submit_convert.sh winter_v1

# 5. Then merge them into training datasets (step 3):
cd ../src && python merge_datasets.py \
    --glob '../datagen/data/shards/features_winter_v1_shard*.npz' --dry-run
#    -> rerun with --start-version <N> to write ../datasets/features_desk_v<N>.npz
```

`submit_datagen.sh <dataset> <num_shards> [max_parallel] [shard_offset]`

- **max_parallel** (default 6) caps concurrently-running shards via the array
  `%` throttle. The cluster allows at most 24; **use fewer to leave room for
  your primary research jobs** — e.g. `./submit_datagen.sh winter_v1 7 2`.
- **shard_offset** (default 0) is the first shard id. With the default random
  opening order you rarely need it — re-running already samples fresh openings.
  It mainly matters for `ORDER=sequential` (to get a disjoint slice on a later
  run), or if you want shard filenames in a given run to start at a specific id.

## How it works

- **Resources per shard:** 32 CPU cores, 1 A100 GPU (unused by Winter, but this
  cluster only has GPU partitions), 16 GB RAM, `3:55:00` wall — under the 4h
  limit. 30 games run in parallel (`Threads=1` each), comfortably inside 32 cores.
- **Opening selection (random, default):** each shard shuffles the entire
  921,600-position book with its own RNG seed and plays through it, so it draws
  a different random subset. The seed is derived from the **SLURM array job id**
  (unique per submission) mixed with the shard id, so:
  - shards within one submission sample disjoint-by-chance, different subsets;
  - **every time you re-run `submit_datagen.sh` you get a fresh random sample
    automatically** — no offsets to manage.

  Each opening is played exactly once (`-games 1`, no color-reversed replay —
  that variance-reduction trick is for Elo testing, not datagen). Because a 4h
  shard plays only ~40k of 921,600 openings, accidental overlap within a run is
  rare. To reproduce an exact previous sample, pin `SEED_OVERRIDE=<int>` (shard
  *i* then uses seed `SEED_OVERRIDE + i`).

  Set `ORDER=sequential` instead for *guaranteed* non-overlapping contiguous
  slices (`start = SHARD_ID * OPENINGS_PER_SHARD`); this is fully reproducible
  but reuses the same openings across runs with the same shard ids, so you'd
  bump the `shard_offset` arg between runs.
- **Clean cutoffs:** fastchess is run under `timeout -s INT` ending ~5 min
  before the wall limit, and it writes a game only once finished, so shards are
  never truncated mid-game. `-recover` restarts a crashed
  engine instead of aborting the shard.
- **Throughput:** ~40k games per shard was measured locally at 3+0.03; expect a
  similar order on the cluster. So ~250k games ≈ 6–7 shards.

## Adjudication

Currently set to match your previous local runs (in `datagen_config.sh`):

```
DRAW_ADJ="movenumber=50 movecount=10 score=5"   # draw after move 50 if both
                                                # evals stay within ±5cp for 10 moves
RESIGN_ADJ=""                                    # resign adjudication OFF
```

**Recommendation:** the draw rule above is conservative and safe to keep. If you
later want a meaningful speedup, consider enabling a *conservative* two-sided
resign rule — it ends clearly-decided games early with very low mislabeling risk
because both sides must agree the position is lost:

```bash
RESIGN_ADJ="movecount=5 score=1000 twosided=true" ./submit_datagen.sh winter_v1 7
```

(`score=1000` cp ≈ +10 pawns sustained for 5 moves by both engines.) Leave it
off if you specifically want full WDL fidelity in long won/lost endgames.

## Common overrides

Any config value can be set inline. Examples:

```bash
# Gentler still on the cluster: 16 games in parallel instead of the default 24
CONCURRENCY=16 ./submit_datagen.sh winter_v1 7 2

# Different time control
TC=5+0.05 ./submit_datagen.sh winter_fast 7

```

## Output layout

```
data/raw/winter_v1_shard00000.pgn   # one PGN per shard (with eval comments)
data/raw/winter_v1_shard00001.pgn
...
data/shards/features_winter_v1_shard00000.npz   # per-shard dataset (step 2)
data/shards/targets_winter_v1_shard00000.npz
data/shards/winter_v1.manifest      # shard list driving the conversion array
logs/winterdata_<jobid>_<task>.out  # per-shard datagen stdout/err
logs/winterconv_<jobid>_<task>.out  # per-shard conversion stdout/err

../datasets/features_desk_v318.npz  # merged training datasets (step 3)
```

## Deployment

These scripts are **not** tracked in git — they carry the cluster account, partition
and filesystem paths. They live inside the repository only so that
`submit_convert.sh` can resolve `../src/pgn_to_dataset.py`; `/datagen/` is in the
repository's `.gitignore`.

Deploy by syncing this directory into a checkout on the cluster:

```bash
rsync -av --exclude data/ --exclude logs/ --exclude '*.epd' \
      ~/Claude/git/WinterTraining/datagen/ \
      <cluster>:~/WinterTraining/datagen/
```

The portable half of the pipeline — `src/game_filter.py`, `src/pgn_to_dataset.py`,
`src/merge_datasets.py` — is tracked and comes from `git pull` on the cluster side.
The binaries (`fastchess`, `Winter`) and the opening book are excluded above and
copied once by hand.
