#!/usr/bin/env python3
"""Merge per-shard dataset files into training datasets of similar size.

Step 3 of the pipeline: data generation writes many small PGN shards, each shard is
converted independently by pgn_to_dataset.py (which is what keeps a conversion inside the
SLURM wall limit), and this rebuilds those pieces into the handful of evenly sized datasets
that training actually loads.

Naming matters here. loader.discover_dataset_tags globs ``features_desk_v*.npz`` and parses
the tag with ``^(\\d+)([a-z]*)$``, and loader.newest_variants keeps only the newest revision
of each version number -- so ``desk_v318a`` supersedes ``desk_v318`` rather than joining it.
Several outputs from one merge must therefore take consecutive *version numbers*, never
letter suffixes, or all but one would silently disappear from ``--datasets all``. That is
what --start-version allocates. Keep the shards themselves out of the datasets directory
(the default --out-dir is not the default shard location) so they are never discovered as
datasets in their own right.

Examples
--------
    # See how the shards would be split, without writing anything:
    python merge_datasets.py --glob '../datasets_shards/features_run7_shard*.npz' --dry-run

    # Merge into ~16M-row datasets numbered desk_v318, desk_v319, ...:
    python merge_datasets.py --glob '../datasets_shards/features_run7_shard*.npz' \\
                             --start-version 318
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import scipy.sparse

# Matches how pgn_to_dataset.py writes a dataset: a CSR feature matrix saved with
# scipy.sparse.save_npz and an int8 label vector saved with np.savez as 'arr_0'.
FEATURES_PREFIX = "features_"
TARGETS_PREFIX = "targets_"


def targets_path(features_path):
    """The targets file paired with a features file, by the shared naming convention."""
    directory, name = os.path.split(features_path)
    if not name.startswith(FEATURES_PREFIX):
        raise ValueError(f"not a features file: {features_path}")
    return os.path.join(directory, TARGETS_PREFIX + name[len(FEATURES_PREFIX):])


def shard_rows(features_path):
    """Row count of a shard, without loading its data.

    np.load returns a lazy NpzFile, so reading only 'shape' skips the data, indices and
    indptr arrays entirely. That keeps the planning pass cheap even over hundreds of shards.
    """
    with np.load(features_path) as fh:
        shape = fh["shape"]
    return int(shape[0]), int(shape[1])


def plan_groups(shards, rows_per_file, num_files=None):
    """Split ``shards`` into consecutive groups of roughly equal row count.

    Shards are kept in their given order and never split, so a group is a contiguous run.
    The number of groups is chosen first and the target is then recomputed as an even
    share, which avoids the undersized trailing file a plain running-total split produces.
    """
    total = sum(rows for _, rows in shards)
    if num_files is None:
        num_files = max(1, round(total / rows_per_file)) if total else 1
    num_files = min(num_files, len(shards))
    target = total / num_files

    groups = []
    current = []
    current_rows = 0
    for index, (path, rows) in enumerate(shards):
        shards_left = len(shards) - index        # including this one
        groups_left = num_files - len(groups)    # including the one being built
        if current:
            # Closing leaves groups_left - 1 still to fill from shards_left shards, so it
            # is only possible while that many remain -- and becomes mandatory at the point
            # where every remaining shard has to be a group of its own. Without the second
            # case a group that never quite reaches its share swallows the tail and the
            # requested file count is silently missed.
            can_close = shards_left >= groups_left - 1
            must_close = shards_left == groups_left - 1
            # Close on reaching the share, counting the incoming shard half in so that
            # overshooting and undershooting the target are traded off evenly.
            wants_close = groups_left > 1 and current_rows + rows / 2 > target
            if must_close or (wants_close and can_close):
                groups.append((current, current_rows))
                current, current_rows = [], 0
        current.append(path)
        current_rows += rows
    if current:
        groups.append((current, current_rows))
    return groups


def merge_group(paths):
    """Load and concatenate one group of shards into a single (features, targets) pair."""
    features = []
    targets = []
    for path in paths:
        f = scipy.sparse.load_npz(path)
        t = np.load(targets_path(path))["arr_0"]
        if f.shape[0] != t.shape[0]:
            raise SystemExit(f"{path}: {f.shape[0]} feature rows but {t.shape[0]} labels")
        features.append(f)
        targets.append(t)
    return scipy.sparse.vstack(features, format="csr"), np.concatenate(targets)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glob', required=True,
                        help="glob matching the shard FEATURES files; each one's targets_ "
                             "counterpart must sit beside it")
    parser.add_argument('--out-dir', type=str, default="./../datasets/",
                        help="directory to write the merged datasets into")
    parser.add_argument('--base-name', type=str, default="desk_v",
                        help="dataset name stem; the version number is appended to it")
    parser.add_argument('--start-version', type=int,
                        help="version number of the first output; later outputs take "
                             "consecutive numbers. Required unless --dry-run.")
    parser.add_argument('--suffix', type=str, default="",
                        help="revision suffix for every output (lowercase letters), e.g. "
                             "'a' to write desk_v318a. Applied to all outputs; they still "
                             "take distinct version numbers.")
    parser.add_argument('--rows-per-file', type=int, default=16_000_000,
                        help="target rows per merged dataset (default 16000000, about the "
                             "size of desk_v300..v302). Ignored when --num-files is given. "
                             "This also sets peak memory: a group is built with its inputs "
                             "and its output both resident, roughly 2x the output size, so "
                             "16M rows peaks near 3.3 GB.")
    parser.add_argument('--num-files', type=int,
                        help="split into exactly this many datasets instead of sizing by rows")
    parser.add_argument('--force', action='store_true',
                        help="overwrite existing output files instead of refusing")
    parser.add_argument('--dry-run', action='store_true',
                        help="report the split and write nothing")
    args = parser.parse_args()

    if args.start_version is None and not args.dry_run:
        parser.error("--start-version is required unless --dry-run is given")
    # loader._VERSION_RE is ^(\d+)([a-z]*)$: anything else stops the dataset being
    # recognised as a revision of its base version.
    if args.suffix and not re.fullmatch(r'[a-z]+', args.suffix):
        parser.error(f"--suffix must be lowercase letters only, got {args.suffix!r}")

    paths = sorted(glob.glob(args.glob))
    if not paths:
        sys.exit(f"No files matched: {args.glob}")
    for path in paths:
        pair = targets_path(path)
        if not os.path.exists(pair):
            sys.exit(f"Missing targets file for {path}: expected {pair}")

    print(f"Reading shapes from {len(paths)} shard(s)...")
    shards = []
    widths = set()
    for path in paths:
        rows, cols = shard_rows(path)
        widths.add(cols)
        shards.append((path, rows))
    if len(widths) > 1:
        sys.exit(f"Shards disagree on feature width: {sorted(widths)}")
    total = sum(rows for _, rows in shards)
    print(f"{total:,} rows of width {widths.pop()} across {len(shards)} shard(s)\n")

    groups = plan_groups(shards, args.rows_per_file, args.num_files)

    start = args.start_version if args.start_version is not None else 0
    names = [f"{args.base_name}{start + i}{args.suffix}" for i in range(len(groups))]
    print(f"{'dataset':>18s} {'shards':>7s} {'rows':>14s}")
    for name, (paths_in_group, rows) in zip(names, groups):
        print(f"{name:>18s} {len(paths_in_group):7d} {rows:14,d}")
    print()

    if args.dry_run:
        if args.start_version is None:
            print("(names shown from version 0; pass --start-version to fix the numbering)")
        return 0

    outputs = []
    for name in names:
        f_out = os.path.join(args.out_dir, f"{FEATURES_PREFIX}{name}.npz")
        t_out = os.path.join(args.out_dir, f"{TARGETS_PREFIX}{name}.npz")
        if not args.force:
            for path in (f_out, t_out):
                if os.path.exists(path):
                    sys.exit(f"Refusing to overwrite {path} (pass --force to replace)")
        outputs.append((f_out, t_out))

    os.makedirs(args.out_dir, exist_ok=True)
    # One group at a time, so peak memory is one merged dataset plus its inputs rather
    # than the whole campaign.
    for name, (paths_in_group, _), (f_out, t_out) in zip(names, groups, outputs):
        features, targets = merge_group(paths_in_group)
        scipy.sparse.save_npz(f_out, features)
        np.savez(t_out, targets.astype(np.int8))
        print(f"wrote {f_out} and {t_out}  ({features.shape[0]:,} rows)")
        del features, targets

    written = names[0] if len(names) == 1 else f"{names[0]} .. {names[-1]}"
    print(f"\nMerged {len(paths)} shard(s) into {len(groups)} dataset(s): {written}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
