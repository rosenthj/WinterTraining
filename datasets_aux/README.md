# Auxiliary datasets

Staging area for additional datasets. Files here follow the same naming as
`datasets/` — `features_desk_v{tag}.npz` / `targets_desk_v{tag}.npz` pairs —
and a dataset may carry additional sidecar files recording how it was built.

Datasets in this directory are **opt-in only**: they are never picked up by
`--datasets all` (which resolves against `datasets/`), so ongoing training runs
are unaffected by anything added here. To train on one, name it explicitly:

```
python train_net.py --datasets 1-221 --aux-datasets <tag> ...
```

`--aux-datasets` uses the same tag grammar as `--datasets` (resolved against
`--aux-data-dir`, default this directory), except `all` is deliberately
rejected — each auxiliary dataset must be named individually.
</content>
