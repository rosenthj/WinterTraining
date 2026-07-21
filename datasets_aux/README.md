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
- `desk_vNovelPv` — PV-neighbourhood dataset (see `src/novel_pv_dataset.py`): the
  adversarial positions themselves are often evaluated fine — the search was fooled
  by misevaluated *downstream* positions, so this dataset harvests the positions the
  search relies on. Bases restricted to runs with <= 12800 nodes (13,038,050 unique
  positions — nearly the whole corpus, high-node runs hold few samples), ordered by
  the same material-signature round-robin (cap 688, 319,335 bases used). Each base
  contributes: itself, the position after every legal move, every position along
  Winter 4.02's PV (searched at the node count from the run directory name, cold
  hash per base), and the PV after every non-engine move preserving the EGTB-best
  W/D/L class (terminal children never searched). In-check and terminal positions
  excluded, globally deduplicated: 10,000,000 positions, W/D/L
  4,045,472 / 4,532,435 / 1,422,093. Seed 0, generated 2026-07-17.
- `desk_vNovelPv2` — same PV-neighbourhood recipe applied to the second-round
  adversarial corpus in fens_v3: positions where Rinter (Winter 4.02 AdvF, the
  adversarially finetuned engine; binary kept as `Rinter` in the repo root,
  verified to reproduce 120/120 recorded moves) still lost tablebase outcome.
  Sources: all Rinter runs (n400/n1600/n6400, 5-6 piece), 881,930 unique base
  positions, material-signature round-robin (cap 1,924, 318,210 bases used).
  10,000,046 positions, W/D/L 4,084,793 / 4,726,632 / 1,188,621. Labels
  independently re-verified against Syzygy on a 5,000-row sample (0 mismatches).
  One feature-level duplicate pair exists (positions differing only in the
  en-passant square, which the 772-feature encoding does not represent; both
  labeled draw). Seed 0, generated 2026-07-19.
- `desk_vNovelPvU1`..`desk_vNovelPvU4` — same PV-neighbourhood recipe applied
  to fens_v2 (the original Winter corpus) with the material-signature
  balancing turned off (`--no-balance`): bases are processed in one uniform
  random shuffle of all 13,038,050 unique bases and split into four
  sequential 10M-position sets so the material distribution reflects the
  corpus as found (see the material-distribution discussion in this
  session -- the round-robin cap had been discarding 97.5% of the corpus,
  concentrated in exactly the signatures Winter blunders most). Each set is
  deduplicated independently, not against the others; measured overlap is
  small (~2.1% between any two sets, 2.78% of the 40,000,091 total rows are
  cross-set duplicates -- 38,886,631 unique positions overall). No base is
  reused across sets: 1,218,216 of the 13,038,050 bases were consumed (~9.3%
  of the corpus), zero engine errors. Each set has a matching
  `bases_desk_vNovelPvU<i>.csv` sidecar (FEN, engine move, node count) listing
  the exact bases expanded into it, enabling in-sample/out-of-sample analysis
  the way `bases_desk_vNovelPv.csv` (below) does for the balanced set. Labels
  independently re-verified against Syzygy on 5,000-row samples per set (0
  mismatches). Per-set W/D/L counts: set 1
  [3,879,120 / 4,732,079 / 1,388,817], set 4
  [3,822,946 / 4,974,183 / 1,202,889] (sets 2-3 similar). Seed 0, generated
  2026-07-20.

All PV/Bal datasets above also have a `bases_<name>.csv` sidecar recording
exactly which base positions were expanded to produce them (`desk_vNovelPv`,
`desk_vNovelPv2`, `desk_vNovelBal` were reconstructed after the fact from
their deterministic seed-0 generation order, since the original runs
predated this bookkeeping; `desk_vNovelPvU*` recorded it live). This is what
lets you split any later corpus into "bases this dataset trained on" vs
"bases it didn't" for transfer analysis.
