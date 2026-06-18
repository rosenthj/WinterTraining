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

The sparse one-hot encoding is the "compressed format" — storing only the handful of nonzero
entries per 772-dim position is far smaller than dense storage.

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

`--optimizer sgd` (default) uses SGD with `--momentum` (0.9), historically the best performer.
`--optimizer ranger` uses Ranger (RAdam + Lookahead + gradient centralization, from
`ranger.py`) for experimentation. `--weight-decay` applies to either. Ranger typically wants a
much lower learning rate than the SGD-tuned `--init-lr` (Adam scale, ~10–30× smaller).

By default the optimizer is recreated at each LR-schedule step. For Ranger, add
`--persistent-optimizer` so a single optimizer spans the whole run (its LR is set externally at
each step), preserving the RAdam/Lookahead state that Ranger relies on — this is how Ranger is
meant to run. The persistent optimizer's state is checkpointed and restored across resumed
segments like everything else.

```bash
python train_net.py --datasets all --exclude 0 vEnd \
    --optimizer ranger --persistent-optimizer --init-lr 0.002 --min-lr 1e-5
```

Ranger tuning knobs: `--eps` (default `1e-5`), `--beta1` / `--beta2` (defaults `0.95` / `0.999`).
If Ranger's loss *diverges late* (creeps up first, with `train/grad_norm` rising only
afterwards), the cause is the adaptive step blowing up in flat regions as the second moment
shrinks — **raise `--eps`** (try `1e-3`, up to `1e-2`) rather than only lowering the LR, which
merely delays it. `--clip-grad-norm <v>` clips the total gradient norm each step as a safety net
(off by default; `train/grad_norm` still reports the pre-clip value so spikes stay visible —
read it off TensorBoard to pick `v`). Example:

```bash
python train_net.py --datasets all --exclude 0 vEnd --optimizer ranger --persistent-optimizer \
    --init-lr 0.001 --min-lr 1e-6 --eps 1e-3 --clip-grad-norm 4
```

`--optimizer winter_ranger` is a RangerLite-inspired variant (`winter_ranger.py`) — Positive-
Negative Momentum + Stable Weight Decay + Norm Loss + Lookahead — the configuration used by
Stockfish's NNUE trainer, adapted to this codebase (Lookahead is a periodic merge into the live
weights, so checkpoints/serialization use the averaged weights with no `eval()`/`train()` hooks;
its numerics are verified bit-exact against `ranger_lite.py`). Defaults reproduce the full
RangerLite config (`eps=1e-7`, `betas=(0.9, 0.999)`, norm loss + PNM on, `lookahead-k=5`); the
update is Adam-scale, so use `--init-lr ~1e-3`. Component knobs for ablation: `--normloss-factor`
(0 disables), `--no-pnm`, `--lookahead-k`, `--lookahead-alpha`, plus `--weight-decay` (stable,
variance-normalized here).

`--reg-weights-only` applies norm loss and weight decay to the Linear/Conv `.weight` tensors
only, exempting biases and bias-like parameters (e.g. `b1`, the per-(channel,square) bias map).
Recommended for this architecture — norm loss assumes a weight matrix whose first dim indexes
neurons, so regularizing a bias map is not meaningful. It mainly changes `winter_ranger`'s norm
loss (which is on by default); for sgd/ranger it only matters with `--weight-decay`.

```bash
python train_net.py --datasets all --exclude 0 vEnd --optimizer winter_ranger \
    --persistent-optimizer --init-lr 0.001 --min-lr 1e-6
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

Training (`train.py`) optimizes a combined WDL MSE + cross-entropy loss and periodically saves
both `{name}.pt` (PyTorch state dict) and `{name}.bin` (raw little-endian weight buffer consumed
by Winter via `model.serialize`). The older `script.py` (hardcoded dataset list) is kept for
reference but `train_net.py` is the preferred entry point.

### TensorBoard

`train_net.py` logs to TensorBoard by default (under `--tb-dir`, default `../logs/tb/<name>`).
View it with:

```bash
tensorboard --logdir logs/tb
```

Scalars are logged at each `--log-freq` interval and once at every epoch end (so the overhead
is negligible — per-batch loss components are summed on-device and synced only when logging):

- `train/loss` (optimized total), `train/loss_reg` (WDL regression term), `train/loss_ce`
  (cross-entropy term)
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

### Resuming a run

`scheduled_lr_train` decays the learning rate over a fixed schedule (`step = epoch //
epochs_per_step`, `lr = init_lr * lr_mult**step`) and after every epoch saves:

- `../models/{name}/{name}_ep{N}.pt` and `_tmp.pt` — model weights, and
- `../models/{name}/{name}.state.pt` — the schedule position (next epoch + step) and optimizer
  state.

Passing `--auto-resume` to `train_net.py` reloads the newest checkpoint **and** that schedule
state, so training continues exactly where it stopped (correct LR, epoch, and optimizer
momentum) instead of restarting the schedule. Resuming an already-finished run is a no-op.
(`--load <path>` remains a one-off weight load that does *not* resume the schedule.)

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

- `gen_ending_data.py`, `max_entropy_extraction.py`, `move_order_writer.py`, `count.py` —
  auxiliary data-generation / analysis utilities.
- `model.py` — network definitions. The relative-conv family (`NetRel`, `NetRelX`, `NetRelH`,
  `NetRelHD`, …) is what Winter uses; see the section above for the currently deployed one.
</content>
