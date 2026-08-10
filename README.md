# WinterTraining

Resources for training neural networks for the [Winter](https://github.com/rosenthj/Winter) chess engine.

The code lives in `src/`. The workflow has two stages:

1. **Dataset generation** — transform `.pgn` game files into a compact, compressed
   on-disk representation of individual training positions.
2. **Training** — load one or more generated datasets and train a network, exporting
   both a PyTorch checkpoint (`.pt`) and a Winter-readable serialized weights file (`.bin`).

## Requirements

- Python 3
- [`python-chess`](https://python-chess.readthedocs.io/) (`chess`, `chess.pgn`, `chess.syzygy`)
- `numpy`, `scipy`, `torch`
- `tensorboard` (optional; for training metrics — training runs without it if absent)

Syzygy endgame tablebases are required for dataset generation. The code currently expects
them at `../../../Chess/TB_Merged` relative to `src/` (see *Paths* below).

## Directory layout

All scripts are run **from inside `src/`** and use paths relative to it:

| Location | Purpose |
|----------|---------|
| `../pgns/{name}.pgn`        | Input PGN files (not in repo) |
| `../datasets/`              | Generated datasets — `features_*.npz`, `targets_*.npz` (git-ignored) |
| `../models/{name}/`         | Trained model checkpoints (`.pt`) and serialized weights (`.bin`) |
| `../logs/`                  | Training logs |
| `../../../Chess/TB_Merged`  | Syzygy tablebase directory |

### Machine-specific paths

The SLURM submission scripts need two locations that differ per machine: the conda
environment and the Syzygy tablebase directory. Neither is committed. Copy the template
and fill it in once:

```bash
cp local_env.sh.example local_env.sh     # git-ignored
```

`local_env.sh` sets `WINTER_CONDA_ENV` and `WINTER_TB_PATH`, and is sourced by
`generate_dataset.sh`, `standby_train.sh`, `standby_baseline.sh` and
`datagen/datagen_config.sh`. Exported environment variables take precedence, so a one-off
run can override without editing it:

```bash
WINTER_TB_PATH=/other/tb sbatch generate_dataset.sh desk_v312
```

A script that needs a path it cannot find fails immediately with a message naming the
variable, rather than running with a wrong default.

### Pushing to the cluster (`sync_to_cluster.sh`)

Code reaches the cluster by `git pull`. What git deliberately does not track — the
fastchess binary and the DFRC opening book — is pushed by `sync_to_cluster.sh` to
`WINTER_SYNC_DEST` (also set in `local_env.sh`, so no cluster path is committed):

```bash
./sync_to_cluster.sh                       # fastchess + opening book
./sync_to_cluster.sh -n                    # dry run
./sync_to_cluster.sh --code                # also tracked source, for an uncommitted change
./sync_to_cluster.sh src/merge_datasets.py # or just one file, by name
```

The **Winter engine binary is not synced**: a local build does not run on the cluster, so
it has to be compiled there. Build it as `datagen/Winter` or point `ENGINE` at it in
`datagen/datagen_config.sh`. fastchess is a static binary and does travel.

It never deletes at the destination (no `--delete`) and always excludes `.git/`, so the
cluster checkout's history and working state are untouched, as are generated datasets,
PGNs, checkpoints and logs. `local_env.sh` is excluded even under `--code`: the cluster's
copy holds *its* paths, and overwriting them with yours would break every job. `--env`
pushes it anyway if you really mean to.

## Stage 1: PGN → dataset

Run from `src/`:

```bash
cd src
python pgn_to_dataset.py --name merged
```

Options:

- `--name` — base name of the input PGN and generated dataset (default `merged`).
- `--pgn-dir` — directory holding `{name}.pgn` (default `./../pgns/`).
- `--out-dir` — where the generated `.npz` files are written (default `./../datasets/`,
  i.e. the same directory the loaders read from).
- `--tablebase` — path to the Syzygy tablebase directory (default `../../../Chess/TB_Merged`).
- `--tb-relabel-prob` — probability of relabelling an endgame position past the entry into the
  endgame from the tablebase (default `1.0`, i.e. relabel everything). See below.
- `--drop-abnormal` — handle games that ended in a time forfeit (default off): repaired from
  the tablebase where the game reached one, dropped where it did not. Required for
  `desk_v303`–`desk_v317`. See below.
- `--out-suffix` — revision suffix appended to the dataset name but **not** the PGN name, so a
  regenerated dataset can be tagged without renaming or copying its PGN. Lowercase letters only.
  `--name desk_v311 --out-suffix a` reads `../pgns/desk_v311.pgn` and writes
  `features_desk_v311a.npz` / `targets_desk_v311a.npz`. `train_net.py` resolves a version to its
  newest revision (`loader.newest_variants`), so once `desk_v311a` exists both `--datasets 311`
  and `--datasets all` pick it over `desk_v311` with no further changes — and the old dataset can
  stay on disk as a fallback.

This reads `../pgns/merged.pgn` and produces two files in `../datasets/`:

- `features_merged.npz` — a **SciPy sparse CSR matrix** (saved via `scipy.sparse.save_npz`),
  one row per extracted position.
- `targets_merged.npz` — a NumPy archive (`np.savez`) of integer result classes, one per row,
  accessed as `['arr_0']`.

### What gets extracted

For each game (`data.py`):

- **Position sampling** (`extract_fens_from_game`): positions are sampled along the game while
  skipping non-quiet positions and positions shortly after a capture/pawn move (via the
  halfmove clock), with randomized spacing so the dataset isn't dominated by any single game.
- **Feature encoding** (`chess_utils.get_features`): each position becomes a **772-dimensional
  one-hot vector**:
  - `12 × 64 = 768` piece-square features (6 white piece types, then 6 black), plus
  - `4` castling-rights features (W queenside, W kingside, B queenside, B kingside).
  - Positions are standardized to **white-to-move** (the board is mirrored and the result
    flipped if black is to move). Optional horizontal flip (when no castling rights) and
    vertical flip (when no pawns) are applied for augmentation/canonicalization.
- **Result labels** are stored as a class: `0 = white win`, `1 = draw`, `2 = black win`
  (from the white perspective after standardization).
- **Tablebase correction**: for positions with ≤6 pieces and no castling rights, the game
  result is replaced by the exact Syzygy WDL value. The script reports how many results
  were changed by tablebase probes.

  Positions are labelled walking backwards from the end of the game, threading the result
  as it goes. Since the piece count never increases, the tablebase positions form a suffix
  of the game, and the **entry into the endgame** — the earliest position that can be probed
  — is the one whose corrected result propagates back through the whole middlegame. Deeper
  endgame positions only ever relabel themselves.

  `--tb-relabel-prob p` exploits that split. The entry position is always probed, so the
  backward propagation is unchanged, but each deeper endgame position keeps the game's
  **actual outcome** with probability `1 - p` instead of the tablebase value. This retains
  the practical difficulty of actually converting the endgame, which relabelling everything
  erases. As a side effect it also cuts tablebase probes: on `desk_v312`, `p = 0` reduces
  them from ~8.6 per endgame-reaching game to 1.

  Note that Winter itself has no tablebase probing in search, so the net is the only endgame
  evaluator — `p = 0` gives up the exact WDL signal entirely. Intermediate values are the
  interesting regime, though on forfeit-filtered data the game result and the Syzygy verdict
  agree on ~97% of endgame positions, so there is not much for this knob to preserve.

### Time forfeits (`--drop-abnormal`)

Some self-play runs were generated on contended cluster nodes and contain a large number of
games lost on time. fastchess records these by writing the forfeiting side's last move and
then scoring the game against them, so the game stops at a position that is neither mate,
stalemate, insufficient material, a repetition, nor a 50-move draw — which is exactly how
`data.abnormal_final_board` detects them, with no tablebase needed.

The recorded result of such a game is unrelated to the position: across `desk_v312` the
forfeiting side was materially *ahead* about as often as behind, and 116 games in the first
6000 ended with a result exactly inverted relative to the Syzygy verdict.

**Such a game is repaired where possible rather than discarded.** The recorded result is only
ever the seed of the backwards walk in `data_from_fen_res_set`, so if the game reached a
tablebase position the probe at the entry into the endgame overwrites it, and every earlier
position is labelled from the Syzygy verdict instead — which is exactly the behaviour wanted
from a forfeited game. Measured on `desk_v312` and `desk_v315`, the pre-endgame label of such
a game matched the entry verdict in 324 of 324 cases, having overridden the recorded result in
203 of them. A forfeited game that never reached a tablebase position has nothing to override
its result and is dropped. On `desk_v312` that keeps roughly 54% of forfeited games (about 10%
of the corpus that a whole-game drop would discard); on `desk_v315`, where forfeits are far
more often non-endgames, only about 16%.

Because the repair is the only thing making these games usable, they always take every
tablebase probe regardless of `--tb-relabel-prob`.

### Game validation (`game_filter.py`)

`game_filter.check_game` screens every game before extraction, replacing the separate
filtering pass that used to run over the shard PGNs. Two of its checks are not a
data-quality question but a robustness one, and so apply regardless of `--drop-abnormal`:
an unfinished `*` result raises out of `utils.string_to_result_class`, and an unparsable
mainline leaves `game.ItGame.make_move` tripping its own assertion. Either aborts the whole
conversion rather than skipping one game, and data generation runs on the preemptable
`standby` QOS, so a shard killed mid-write is a live scenario.

It also corrects a class of forfeit the positional test cannot detect. When a side flags on
the very move that delivers mate, the final position is a genuine checkmate — nothing about
it looks abnormal — and only the result is wrong. The final position states the result
outright, so these are corrected from it rather than dropped:

| file | games contradicting a terminal final position |
|---|---|
| `desk_v222` | 0.00% |
| `desk_v312` | 0.10% |
| `desk_v307` | 0.12% |
| `desk_v315` | 0.38% |

Every instance found so far is White mating and being scored `0-1`. Left uncorrected they
are the worst labels in the corpus: a full-length, normally played game teaching that a
mated king won.

Measured rates over the first 2000 games of each PGN:

| file | rate | file | rate | file | rate |
|---|---|---|---|---|---|
| `desk_v222`–`v224` | 0.0% | `desk_v305` | 29.6% | `desk_v311` | 22.9% |
| `desk_v300` | 0.0% | `desk_v306` | 7.6% | `desk_v312` | 13.8% |
| `desk_v301` | 1.6% | `desk_v307` | 29.7% | `desk_v313` | 1.0% |
| `desk_v302` | 2.3% | `desk_v308` | 2.2% | `desk_v314` | 0.8% |
| `desk_v303` | 11.7% | `desk_v309` | 12.3% | `desk_v315` | 52.5% |
| `desk_v304` | 9.9% | `desk_v310` | 6.2% | `desk_v316` | 48.1% |
| | | | | `desk_v317` | 4.0% |

The rate tracks node contention rather than any tournament setting — `desk_v313` and
`desk_v314` were generated alongside `desk_v315` and `desk_v316`. Newer runs handle this at
shard merge time, so the flag is off by default and is only needed when regenerating these
files. It is a no-op on clean PGNs.

Filtering also removes most of what looked like endgame conversion failures. On `desk_v312`,
the share of endgame positions where the game result differs from the Syzygy verdict falls
from 14.6% to 2.8% at six pieces and from 18.0% to 2.7% at five, and the conversion rate of
tablebase-won endgame entries rises from 91.5% to 98.0%.

The sparse one-hot encoding is the "compressed format" — storing only the handful of nonzero
entries per 772-dim position is far smaller than dense storage.

### Merging shards (`merge_datasets.py`)

Data generation (see `datagen/`) writes many small PGN shards rather than a few large files.
Converting each shard independently is what keeps a conversion comfortably inside the SLURM
wall limit — a 475k-game PGN takes about 1.8 hours of local-equivalent work against a 3:55
window, where a ~40k-game shard takes about 9 minutes. `merge_datasets.py` then rebuilds the
per-shard `.npz` pieces into the handful of evenly sized datasets that training loads:

```bash
# See the split first; writes nothing.
python merge_datasets.py --glob '../datasets_shards/features_run7_shard*.npz' --dry-run

# Merge into ~16M-row datasets numbered desk_v318, desk_v319, ...
python merge_datasets.py --glob '../datasets_shards/features_run7_shard*.npz' \
                         --start-version 318
```

Each shard's `targets_` file must sit beside its `features_` file; only the features glob is
given. Shards are never split, so groups are contiguous runs of whole shards, and the group
count is chosen before the target is recomputed as an even share — which avoids the
undersized trailing file a plain running-total split produces. Use `--num-files` to fix the
count directly, `--rows-per-file` to size by rows (this also sets peak memory: a group is
built with its inputs and its output both resident, so 16M rows peaks near 3.3 GB), and
`--force` to overwrite. Existing outputs are refused otherwise.

**Outputs take consecutive version numbers, never letter suffixes.** `newest_variants` keeps
only the newest revision of each version, so writing `desk_v318`/`desk_v318a` from one merge
would make all but one silently vanish from `--datasets all`. `--start-version` allocates the
numbers; `--suffix` applies one revision letter across all of them, which still leaves their
numbers distinct. Keep the shards themselves outside `../datasets/` so
`discover_dataset_tags` never picks them up as datasets in their own right.

## Stage 2: Training

The training entry point is **`train_net.py`**, which selects datasets from the command line
(no need to edit a hardcoded list as the old `script.py` required):

```bash
cd src
python train_net.py --name my_model --datasets all --exclude vEnd \
    --batch-size 256 --init-lr 0.008 --min-lr 0.0001 --epochs-per-step 2
```

### Selecting datasets (`--datasets`)

Datasets live in `--data-dir` (default `../datasets/`) as `features_desk_v{tag}.npz` /
`targets_desk_v{tag}.npz` pairs. `--datasets` takes any mix of:

- version tags — `5`, `100a`
- inclusive ranges — `200-221`
- special names — `vEnd`
- `all` — every numeric version

The **newest variant always wins**: if both `v100` and `v100a` exist, selecting either `100`
or `100a` loads `v100a` (the `a` revision supersedes the plain one). A leading `v` is optional.
`--exclude` drops datasets using the same grammar; `--list` prints the resolved selection and
exits without training. Examples:

```bash
python train_net.py --datasets 200-221            # one numbered range
python train_net.py --datasets 2 5 100a vEnd      # explicit mix
python train_net.py --datasets all --exclude 50 51 vEnd
python train_net.py --datasets all --list         # preview, don't train
```

### Auxiliary datasets (`--aux-datasets`)

Datasets in `--aux-data-dir` (default `../datasets_aux/`, see its README) are **opt-in
only**: `--datasets all` never touches them, so experimental datasets can be staged there
without affecting other training runs. `--aux-datasets` uses the same tag grammar, except
`all` is rejected — each auxiliary dataset must be named individually:

```bash
python train_net.py --datasets 1-221 --aux-datasets <tag>
```

Other useful flags: `--portion` (subsample each dataset), `--val-name` (validation set, default
`validation_games`), `--model`/`--d`/`--fd`/`--num-inputs` (architecture; defaults reproduce the
deployed `NetRelHD(d=16, fd=64, num_inputs=768)`), `--load <ckpt>`, `--device N`, `--no-cuda`,
`--lr-mult`, `--log-freq`.

### Large datasets that don't fit in memory (`--reload-every`)

Loading every dataset at once concatenates them into a single in-memory sparse matrix, which
can exceed the job's RAM (e.g. all 43 desk datasets at `--mem=16g`). To bound memory, use
resampling:

```bash
python train_net.py --datasets all --exclude 0 vEnd --reload-every 1 --portion 0.25
```

With `--reload-every N`, the trainer draws a **fresh random `--portion` subset every N epochs**
instead of holding the whole corpus. Only ~`portion` of the data is resident at any time, but
because a new subset is drawn each reload, training still covers all of it over the run. This
revives the old `train_v2` streaming behaviour. `--reload-every 0` (default) loads once, as
before. (Per-dataset fractions are also possible via the `load_from_multiple` tuple API.)
Alternatively, raise `#SBATCH --mem` in `standby_train.sh` and keep `--portion 1.0`.

### Optimizer

`--optimizer sgd` (the default, and the only option) uses SGD with `--momentum` (0.9), which has
been the best performer here. `--weight-decay` applies on top of it. Adaptive optimizers of the
Ranger family were tried and did not improve on SGD, so they have been removed.

By default the optimizer is recreated at each LR-schedule step. `--persistent-optimizer` instead
builds one optimizer that spans the whole run (its LR is set externally at each step), carrying
the momentum buffers across LR drops rather than resetting them. Its state is checkpointed and
restored across resumed segments like everything else. WSD runs (`--schedule wsd`) always use a
persistent optimizer.

`--clip-grad-norm <v>` clips the total gradient norm each step as a safety net (off by default;
`train/grad_norm` still reports the pre-clip value so spikes stay visible — read it off
TensorBoard to pick `v`).

`--reg-weights-only` applies weight decay to the Linear/Conv `.weight` tensors only, exempting
biases and bias-like parameters (e.g. `b1`, the per-(channel,square) bias map). It only matters
in combination with `--weight-decay`.

```bash
python train_net.py --datasets all --exclude 0 vEnd \
    --persistent-optimizer --init-lr 0.008 --min-lr 1e-4 --clip-grad-norm 4
```

Loading helpers in `loader.py`:

- `load_features_results(name)` / `load_dataset(name)` — load a single `features_{name}.npz` +
  `targets_{name}.npz` pair.
- `load_from_multiple([...])` — concatenate several `features_desk_v{tag}.npz` /
  `targets_desk_v{tag}.npz` datasets, with optional per-dataset subsampling (`portion`).
- `discover_dataset_tags(dir)` / `select_dataset_tags(tokens, available)` — the dataset
  discovery and newest-variant selection used by `train_net.py`.
- `make_scatter_loader(features, results, batch_size, shuffle, device)` — wraps the data in a
  `ScatterLoader`, which densifies each one-hot batch directly on the training device (only the
  active column indices cross to the GPU, not full 772-wide dense rows).

Training (`train.py`) optimizes a combined loss `total = reg + ce_weight · ce + draw_weight ·
draw`, where `reg` is the WDL regression term (MSE on the win-minus-loss probability), `ce` is
the W/D/L cross-entropy, and `draw` is the draw-axis MSE (`MSE(p_draw, draw_indicator)`).
`--ce-weight` (default `0.04`, the historical value) sets the cross-entropy weight: the
regression term is blind to the draw axis (moving probability symmetrically between win and loss
leaves the win-minus-loss margin unchanged), so raising `--ce-weight` puts more pressure on full
W/D/L calibration — watch its effect on `val/wasserstein`.

`--draw-weight` (default `0`, off) adds the bounded draw-axis MSE term as an alternative draw
calibrator to the log-barrier cross-entropy. It is on the same squared-error scale as `reg`, and
`reg + 3 · draw` equals twice the full multiclass Brier score, so **`--draw-weight 3` (with
`--ce-weight 0`) reproduces the Brier score** — draw calibration without a separate `--loss`
enum. Both `ce` and `draw` are proper scoring rules for the draw axis and are asymptotically
redundant, so the usual setup is `reg` plus *one* of them; the narrow reason to combine all three
is `draw` to calibrate the bulk draw rate plus a small `ce` as a tail/overconfidence guard, since
they penalize the draw region differently. The `draw` term is always computed and logged as
`train/loss_draw` even when its weight is 0, so it is a free diagnostic; when turning up
`--draw-weight`, watch `train/loss_reg` — since the two share the same scale, a large draw weight
pulls capacity off the eval axis. It saves a PyTorch
checkpoint (`.pt`) for the always-latest `{name}_tmp` and for every per-epoch snapshot
`{name}_ep{N}`. The Winter-readable serialized weights (`.bin`, a raw little-endian buffer via
`model.serialize`) are written **only for `{name}_tmp.bin`** — the most up-to-date model, kept
as a convenience for loading into the engine. To get a `.bin` for a specific epoch snapshot,
re-serialize its `.pt` with `model.serialize`. The older `script.py` (hardcoded dataset list) is
kept for reference but `train_net.py` is the preferred entry point.

### TensorBoard

`train_net.py` logs to TensorBoard by default (under `--tb-dir`, default `../logs/tb/<name>`).
View it with:

```bash
tensorboard --logdir logs/tb
```

Scalars are logged at each `--log-freq` interval and once at every epoch end (so the overhead
is negligible — per-batch loss components are summed on-device and synced only when logging):

- `train/loss` (optimized total), `train/loss_reg` (WDL regression term), `train/loss_ce`
  (cross-entropy term), `train/loss_draw` (draw-axis MSE term; logged even when
  `--draw-weight 0`, as a free calibration diagnostic)
- `train/lr` — logged densely so it renders as a step function rather than a linearly
  interpolated ramp
- `train/grad_norm` — total gradient L2 norm (a per-log-point snapshot; watch for spikes /
  instability)
- `train/positions_per_sec` — throughput (useful for spotting I/O cost from `--reload-every`)
All of the following come from **one validation pass** at each `--log-freq` interval (and at
every epoch end) — the same read-only `test()` pass, so the activation metrics add no extra
forward and have no effect on training:

- `val/mse`, `val/l1`, `val/accuracy` — WDL regression losses and the fraction of positions
  whose argmax W/D/L class matches the result (random baseline ≈ 0.333).
- `val/wasserstein` — Wasserstein-1 (earth-mover) distance between the predicted W/D/L
  distribution and the actual outcome, on the expected-score axis (W = 1.0, draw = 0.5,
  L = 0.0). Since the target is a single outcome this is just `E_p|score − true_score|`.
  Unlike `val/mse`/`val/l1`, which only see the win-minus-loss margin and are **blind to the
  draw axis**, this is sensitive to where the *draw* probability mass sits, so it reflects W/D/L
  (especially draw) calibration. It's a pure validation diagnostic, independent of `--ce-weight`,
  so it stays comparable across runs with different loss weightings.
- `act/{conv,fc}_frac_zero`, `act/{conv,fc}_frac_max` — *per-element* fraction of clipped-ReLU
  activations pinned at 0 or at the max (8). A saturation/sparsity signal, **not** a dead-neuron
  count: `conv_frac_zero` is dominated by the piece-presence mask (empty squares are structurally
  0), and even `fc_frac_zero` is high simply because activations are sparse.
- `act/conv_active_frac_zero` — `conv_frac_zero` corrected to exclude the mask: the zero fraction
  among only the *computed* (non-masked, piece-occupied) conv activations. The masked fraction is
  read off the input piece planes (`1 − pieces/768`, exact because the mask density is preserved
  through the conv/mirror/cat), so this isolates genuine activation clamping from empty-square
  structural zeros — the meaningful per-element conv saturation signal.
- `val/{conv,fc}_dead_zero`, `val/{conv,fc}_dead_max` — *per-neuron* fraction that is **dead
  across the entire validation set**: a conv channel / fc unit that never rises above 0 (dead at
  zero) or is always pinned at the max (dead at max). This is the true wasted-capacity metric —
  aggregating per-neuron over all positions removes the masking confound, so `conv_dead_zero` is
  meaningful here too. ~0 is healthy (the deployed net is exactly 0).
- `startpos/score`, `startpos/win`, `startpos/draw`, `startpos/loss` — the network's predicted
  expected score / WDL for the opening position (per epoch); a quick interpretable sanity check
  that should settle near a small white advantage

The x-axis (`global_step` = batches seen) is persisted in the run state, so charts stay
continuous across resumed segments. Pass `--no-tensorboard` to disable; if the `tensorboard`
package isn't installed, training continues without it. (Install with `pip install tensorboard`.)

### Learning-rate schedule (`--schedule`)

Two LR schedules are available via `--schedule`:

- **`step`** (default) — the historical geometric decay: `lr = init_lr · lr_mult**(epoch //
  epochs_per_step)`, run until `lr < min_lr`. Kept unchanged so older runs reproduce exactly;
  `--lr-mult` and `--epochs-per-step` control it.
- **`wsd`** — **Warmup–Stable–Decay**, the modern default for training on large datasets with
  SGD. Over a fixed `--total-epochs` budget it (1) linearly **warms up** the LR from ~0 to the
  peak `--init-lr` over `--warmup-steps` batches, (2) holds it **stable** at the peak through the
  long middle of the run, then (3) **decays** it with a half-cosine to `--min-lr` over the final
  `--decay-frac` of the run (default `0.1`). It uses one persistent optimizer so momentum is
  continuous through the stable→decay transition.

Why WSD here: the long high-LR stable phase keeps the network learning **rare features** (with
SGD a weight for a rare piece configuration only updates on the few steps that feature is active,
so decaying the LR early — as geometric step decay does — starves it), while most of the final
loss drop comes from the short end-decay. Warmup avoids the early large, ill-conditioned-gradient
instability. Warmup is measured from **this run's** `global_step 0`, so a fresh or **grown** run
(see `--init-partial` above) automatically re-warms at its start — the right thing after adding
capacity — while a resumed segment of the same run does not warm up again. WSD is also a natural
fit for the chained SLURM segments below: the stable phase is a flat LR you can spread across any
number of ~4h segments, and the decay lands in the final segments. Example:

```bash
python train_net.py --name rew502_wsd --schedule wsd --total-epochs 60 \
    --warmup-steps 2000 --decay-frac 0.15 --init-lr 0.016 --min-lr 0.0002 \
    --datasets all --exclude 0 vEnd --reload-every 1 --portion 0.14 --batch-size 64
```

`train/lr` in TensorBoard shows the warmup ramp, stable plateau, and cosine tail; `train/loss_reg`
should keep improving through the stable phase and drop further during the decay.

### Resuming a run

`scheduled_lr_train` computes the learning rate from a schedule (`step` or `wsd`, above) and after
every epoch saves:

- `../models/{name}/{name}_ep{N}.pt` and `_tmp.pt` — model weights, and
- `../models/{name}/{name}.state.pt` — the schedule position (next epoch, step, and
  `global_step`) and optimizer state.

Passing `--auto-resume` to `train_net.py` reloads the newest checkpoint **and** that schedule
state, so training continues exactly where it stopped (correct LR, epoch, and optimizer
momentum) instead of restarting the schedule. Both schedules resume bit-for-bit: `step` LR is a
function of the epoch, and `wsd` LR is a function of the persisted `global_step`, so a run cut off
by the wall-clock and resumed reproduces the uninterrupted run (verified). For `wsd`, pass the
**same `--total-epochs`** to every segment — it defines the schedule's horizon, so changing it
between segments reshapes the (already-traversed) curve. Resuming an already-finished run is a
no-op. (`--load <path>` remains a one-off weight load that does *not* resume the schedule.)

### Fine-tuning / retraining from an existing model (`--init-from`)

`--init-from <checkpoint.pt>` seeds a **fresh** run's weights from an existing model and then
trains a new schedule — for retraining or continued training rather than starting from random
init. The architecture flags must match the checkpoint (for the deployed net the defaults
already do: `NetRelHD(d=16, fd=64, num_inputs=768)`).

It is designed to compose with `--auto-resume` and the segment chain: an existing checkpoint for
the run takes precedence, so **segment 1 seeds from the base model and later segments resume the
fine-tune run's own checkpoints** (the base is not reloaded each segment). Example — retrain the
current Winter model and check strength is preserved:

```bash
./submit_chain.sh 13 retrain_baseline --datasets all --exclude 0 vEnd --reload-every 1 \
    --portion 0.25 --batch-size 64 --epochs-per-step 6 --init-lr 0.008 \
    --init-from ../models/rn16HD64b.pt
```

Watch `val/mse` (should stay near the deployed net's **0.3075**) and `val/accuracy` (≈0.679).
Note these are *proxies*: true playing strength must be confirmed by loading the resulting
`.bin` into Winter and running an engine match against the original. Lower `--init-lr` keeps the
weights closer to the base (more faithful retraining); higher LR explores further.

#### Growing the net (`--init-partial`)

By default `--init-from` (and `--load`) require the checkpoint's architecture to match exactly.
Adding `--init-partial` relaxes this so a **larger** model can be seeded from a **smaller**
checkpoint — useful when a bigger dataset justifies more capacity. For every parameter shared
between the checkpoint and the new model it copies the overlapping leading block
(`src[:d0, :d1, …]`); the newly added units keep their normal random init, and any layer that is
missing from the checkpoint or can't be aligned (shrinking / rank change) is left untouched. The
load prints a `copied / grown / skipped` summary per parameter. Example — grow the deployed net
to `d=24, fd=96`:

```bash
python train_net.py --name winter_big --d 24 --fd 96 \
    --init-from ../models/rn16HD64b.pt --init-partial \
    --datasets all --exclude 0 vEnd --init-lr 0.008
```

Treat a grown run as a fresh schedule (start from a normal `--init-lr`): the optimizer state
does not carry over and the added units need to train up. `--auto-resume` never partial-loads —
once the run has its own checkpoints they continue at the grown size as usual, so `--init-partial`
composes with the segment chain exactly like `--init-from`.

## Running on a batch cluster (SLURM)

The standby queue caps a job at 4 hours, so a long training run is split into chained
~4-hour segments. Two scripts in the repo root handle this:

- **`standby_train.sh`** — a single SLURM segment. It activates the conda env, `cd`s into
  `src/`, and runs `train_net.py --name <run> --auto-resume <your args>`. Because of
  `--auto-resume`, every segment continues the previous one's weights and LR schedule (a
  no-op on the first segment, and on a segment cut short by the wall-clock limit it resumes
  from the last completed epoch).
- **`submit_chain.sh`** — enqueues N segments as a dependency chain, each held until the
  previous finishes:

  ```bash
  ./submit_chain.sh <num_segments> <run_name> [train_net.py args...]
  # e.g.
  ./submit_chain.sh 6 baseline --datasets all --exclude vEnd --batch-size 256
  ```

  Every segment after the first is submitted with `--dependency=afterany:<prev_jobid>`.
  `afterany` (not `afterok`) is deliberate: a segment that hits the 4h limit or is preempted
  exits non-zero, and the chain must continue anyway. The trade-off is that a genuinely broken
  run also keeps marching — watch the first segment's log under `logs/`.

Monitor with `squeue -u $USER --name=wintertrain`; cancel the whole chain with
`scancel --name=wintertrain -u $USER`. Adjust the `#SBATCH` account/partition lines in
`standby_train.sh` to match your allocation.

## The model currently used by the Winter engine

The Winter engine's `net_evaluation.cc` loads `rn16HD64b.bin` (via `INCBIN`). That network is
produced by the **`NetRelHD`** class in `src/model.py`, instantiated as:

```python
NetRelHD(d=16, fd=64, num_inputs=768, activation=nn.Hardtanh(min_val=0, max_val=8))
```

Decoding the file name `rn16HD64b`: `rn` = relative net, `16` = `d=16`, **`HD`** = **H**idden
full layer + mirror-**D**oubled, `64` = `fd=64`, `b` = revision.

How this was determined:

- `NetRelHD` is the only class combining both the **relative-conv path** (`c1`/`b1`/`out`,
  matching the engine's `net_input_weights` / `bias_layer_one` / `output_weights`) and a
  **full hidden-layer path** (`f1`/`fout`, matching `full_layer_weights` / `full_output_weights`).
- It is also the only such class that is **mirror-doubled** (`out = Conv2d(2*12*d, …)`,
  `fout = Linear(2*fd, …)`, position concatenated with its vertical mirror). The engine
  confirms this: `init_weights()` calls only the `init_mirrored_*` loaders, which read
  half-width weights from the file (`block_size/2`, `full_block_size/2`) and reconstruct the
  mirrored half at load time.
- The engine load order (`init_weights`) matches `NetRelHD.serialize` byte-for-byte:
  `c1` → `b1` → `out` (+bias) → `f1` (+bias) → `fout` (no bias).
- Engine sizes `block_size = 32 = 2·d` and `full_block_size = 128 = 2·fd` give `d=16`, `fd=64`.
  The full layer reads only the `12×64 = 768` piece-square inputs (no castling features), so
  `num_inputs = 768`. The clipped-ReLU at 8 (`clipped_relu(8)`) corresponds to `Hardtanh(0, 8)`.

### Reference performance (held-out `validation_games`, 42,020 positions)

These are the bar new training runs are judged against (the checkpoint `models/rn16HD64b.pt`
loads into the architecture above with no missing/unexpected keys):

| Metric | Value |
|--------|-------|
| WDL-MSE (`val/mse`) | 0.3075 |
| WDL-L1 (`val/l1`)   | 0.3892 |
| Argmax W/D/L accuracy (`val/accuracy`) | 0.679 |
| Start-position eval | W=0.295, D=0.483, L=0.222 → expected score 0.536 |

Activation health (a useful target — a well-trained net wastes no capacity):

| Statistic | conv | fc |
|-----------|------|----|
| Neurons dead-at-zero over the whole val set (never > 0) | 0.000 | 0.000 |
| Neurons dead-at-max over the whole val set (always = 8)  | 0.000 | 0.000 |
| Per-element activations clamped at 0 (sparsity)          | 0.998 | 0.840 |

Note the contrast: per-element saturation is high (activations are sparse), yet **no neuron is
truly dead** — every conv channel and fc unit fires on some positions. So high `act/*_frac_zero`
during training is sparsity, not death; the `val/*_dead_*` metrics (per-neuron, whole val set)
are what actually indicate wasted capacity, and ~0 is healthy.

## Other scripts

- `merge_datasets.py` — merges per-shard datasets into evenly sized ones (see Stage 1).
- `game_filter.py` — game-level validation used by `pgn_to_dataset.py` (see Stage 1).
- `gen_ending_data.py`, `max_entropy_extraction.py`, `move_order_writer.py`, `count.py` —
  auxiliary data-generation / analysis utilities.
- `model.py` — network definitions. The relative-conv family (`NetRel`, `NetRelX`, `NetRelH`,
  `NetRelHD`, …) is what Winter uses; see the section above for the currently deployed one.
