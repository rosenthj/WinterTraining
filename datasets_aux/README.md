# Auxiliary datasets

Datasets in this directory are **opt-in only**: they are never picked up by
`--datasets all` (which resolves against `datasets/`), so ongoing training runs
are unaffected by anything added here. To train on one, name it explicitly:

```
python train_net.py --datasets 1-221 --aux-datasets vNovelNc ...
```

`--aux-datasets` uses the same tag grammar as `--datasets` (resolved against
`--aux-data-dir`, default this directory), except `all` is deliberately rejected.

Current contents (from the GFN novel-search engine-mistake positions, see
`src/novel_fens_to_dataset.py`):

- `desk_vNovel` — 25,556,803 unique positions before/after Winter's tablebase-losing
  moves in 5-6 piece endgames (all Winter novel_mutate runs in fens_v2), deduplicated
  on the side-to-move-mirrored position, labeled by Syzygy WDL. Generated 2026-07-08.
- `desk_vNovelNc` — the same minus the 3,106,869 positions where the side to move is
  in check (static eval is never called in check): 22,449,934 positions.
- `desk_vNovelBal` — material-balanced neighbourhood dataset (see
  `src/novel_balanced_dataset.py`): the 13.2M unique adversarial positions were grouped
  into 567 material signatures (side-to-move perspective) and sampled round-robin
  (rare signatures complete, common ones capped at 1,315), and each of the 561,850
  sampled base positions expanded into itself, the position after every legal move,
  and one guarded random one-piece perturbation of the base and after-engine-move
  positions. In-check and terminal positions excluded, globally deduplicated:
  10,000,013 positions, W/D/L 4,709,485 / 4,463,674 / 826,854. Seed 0,
  generated 2026-07-13.
