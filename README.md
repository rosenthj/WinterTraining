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

Datasets are loaded from `../datasets/` by `loader.py`. The main training entry point is
`script.py`:

```bash
cd src
python script.py --name my_model --batch-size 256 --init-lr 0.008 --min-lr 0.0001 --epochs-per-step 2
```

Useful flags: `--device N` (CUDA device index), `--no-cuda`, `--load <path>` (load an existing
checkpoint instead of training), `--log-freq`.

Loading helpers in `loader.py`:

- `load_features_results(name)` / `load_dataset(name)` — load a single `features_{name}.npz` +
  `targets_{name}.npz` pair.
- `load_from_multiple([2, 5, 6, ...])` — concatenate several `features_desk_v{n}.npz` /
  `targets_desk_v{n}.npz` datasets, with optional per-dataset subsampling (`portion`).

Training (`train.py`) optimizes a combined WDL MSE + cross-entropy loss and periodically saves
both `{name}.pt` (PyTorch state dict) and `{name}.bin` (raw little-endian weight buffer consumed
by Winter via `model.serialize`).

## Other scripts

- `gen_ending_data.py`, `max_entropy_extraction.py`, `move_order_writer.py`, `count.py` —
  auxiliary data-generation / analysis utilities.
- `model.py` — network definitions (`Net`, `NetRel`).
</content>
