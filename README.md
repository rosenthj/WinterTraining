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

Scalars logged at each `--log-freq` interval (so the overhead is negligible — the per-batch
loss components are summed on-device and synced only when logging): `train/loss` (the optimized
total), `train/loss_reg` (WDL regression term) and `train/loss_ce` (cross-entropy term), plus
`train/lr` and per-epoch `val/mse` / `val/l1`. The x-axis (`global_step` = batches seen) is
persisted in the run state, so charts stay continuous across resumed segments. Pass
`--no-tensorboard` to disable; if the `tensorboard` package isn't installed, training continues
without it. (Install with `pip install tensorboard`.)

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

## Other scripts

- `gen_ending_data.py`, `max_entropy_extraction.py`, `move_order_writer.py`, `count.py` —
  auxiliary data-generation / analysis utilities.
- `model.py` — network definitions. The relative-conv family (`NetRel`, `NetRelX`, `NetRelH`,
  `NetRelHD`, …) is what Winter uses; see the section above for the currently deployed one.
</content>
