import glob
import os
import re

import numpy as np
import scipy
import torch
import random

from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler


# Matches a numeric dataset version with an optional lowercase revision suffix, e.g.
# "100" -> (100, ""), "100a" -> (100, "a"). Non-matching tags (e.g. "End") are treated
# as standalone "special" datasets that are never deduplicated.
_VERSION_RE = re.compile(r'^(\d+)([a-z]*)$')


def discover_dataset_tags(data_dir, prefix="features_desk_v", suffix=".npz"):
    """Return the version tags of every desk dataset present in ``data_dir``.

    A tag is the part of the filename between ``prefix`` and ``suffix`` -- e.g.
    ``features_desk_v100a.npz`` yields the tag ``"100a"``.
    """
    tags = []
    for path in sorted(glob.glob(os.path.join(data_dir, f"{prefix}*{suffix}"))):
        name = os.path.basename(path)
        tags.append(name[len(prefix):-len(suffix)])
    return tags


def newest_variants(tags):
    """Collapse a list of tags to the newest variant of each numeric version.

    Returns ``(numeric, specials)`` where ``numeric`` maps an integer base version to
    ``(suffix, tag)`` for the newest variant (a longer/greater suffix is newer, so
    ``100a`` supersedes ``100``), and ``specials`` maps each non-numeric tag to itself.
    """
    numeric = {}
    specials = {}
    for tag in tags:
        m = _VERSION_RE.match(tag)
        if m:
            base, suffix = int(m.group(1)), m.group(2)
            current = numeric.get(base)
            if current is None or suffix > current[0]:
                numeric[base] = (suffix, tag)
        else:
            specials[tag] = tag
    return numeric, specials


def _strip_v(token):
    return token[1:] if token.startswith("v") else token


def select_dataset_tags(tokens, available_tags, exclude=None, warn=print):
    """Resolve CLI selection ``tokens`` against the datasets actually present.

    Each token may be:
      * ``all``            -- every numeric version (newest variant of each);
      * a range ``200-221`` -- every numeric version in the inclusive range;
      * a version ``5`` / ``100a`` -- that version, always resolved to its newest
        variant (so ``100`` and ``100a`` both select ``100a``);
      * a special name ``End`` / ``vEnd`` -- a non-numeric dataset.

    A leading ``v`` is optional on any token. Missing datasets are warned about and
    skipped. The result preserves selection order and removes duplicates and anything
    matched by ``exclude`` (same token grammar).
    """
    numeric, specials = newest_variants(available_tags)

    def resolve(token):
        token = token.strip()
        if token.lower() == "all":
            return [numeric[base][1] for base in sorted(numeric)]
        token = _strip_v(token)
        if "-" in token:
            lo, hi = token.split("-", 1)
            if lo.isdigit() and hi.isdigit():
                out = []
                for i in range(int(lo), int(hi) + 1):
                    if i in numeric:
                        out.append(numeric[i][1])
                    else:
                        warn(f"No dataset found for v{i}, skipping")
                return out
        m = _VERSION_RE.match(token)
        if m:
            base, suffix = int(m.group(1)), m.group(2)
            if base in numeric:
                newest_suffix, tag = numeric[base]
                if suffix and suffix != newest_suffix:
                    warn(f"Requested v{token} but newest variant is v{tag}; using v{tag}")
                return [tag]
            warn(f"No dataset found for v{token}, skipping")
            return []
        if token in specials:
            return [specials[token]]
        warn(f"No dataset found for '{token}', skipping")
        return []

    selected = []
    for token in tokens:
        for tag in resolve(token):
            if tag not in selected:
                selected.append(tag)
    if exclude:
        excluded = set()
        for token in exclude:
            excluded.update(resolve(token))
        selected = [tag for tag in selected if tag not in excluded]
    return selected


def load_features_results(name, data_dir="../datasets/"):
    features = scipy.sparse.load_npz(os.path.join(data_dir, f"features_{name}.npz"))
    results = np.load(os.path.join(data_dir, f"targets_{name}.npz"))['arr_0']
    return features, results


def load_dataset(name, batch_size=16, shuffle=True):
    features, results = load_features_results(name)
    return torch.utils.data.DataLoader(CSRDataset(features,results), batch_size=batch_size, shuffle=shuffle)


def merge_desk(fd, rd, old_tag, new_tag):
    f = scipy.sparse.load_npz(f"features_merged_{old_tag}.npz")
    r = np.load(f"targets_merged_{old_tag}.npz")['arr_0']
    f = scipy.sparse.vstack([f, fd])
    r = np.concatenate([r, rd])
    scipy.sparse.save_npz(f"features_merged_{new_tag}.npz", f)
    np.savez(f"targets_merged_{new_tag}.npz", r)
    return f, r


class CSRDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __getitem__(self, index):
        return np.squeeze(np.asarray(self.features[index].todense())), self.targets[index]

    def __len__(self):
        return self.features.shape[0]


def _identity_collate(batch):
    """The dataset already returns a full batch; keep it as-is."""
    return batch


class BatchIndexCSRDataset:
    """CSR-backed dataset that returns an entire batch at once.

    ``__getitem__`` receives a *list* of row indices (produced by a ``BatchSampler``)
    instead of a single index. It slices those rows out of the CSR matrix in one pass
    and returns the batch as sparse COO components -- the local row index and feature
    column of every nonzero -- plus the targets. Densification is deferred to
    ``ScatterLoader``, which scatters these indices into a dense tensor on the training
    device. Because the features are one-hot, the nonzero *values* are all 1 and never
    need to be stored or transferred.
    """

    def __init__(self, features, targets):
        self.features = features.tocsr()
        self.targets = np.asarray(targets)
        self.num_features = self.features.shape[1]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, batch_indices):
        coo = self.features[batch_indices].tocoo()
        rows = torch.from_numpy(coo.row.astype(np.int64))
        cols = torch.from_numpy(coo.col.astype(np.int64))
        targets = torch.as_tensor(self.targets[batch_indices], dtype=torch.long)
        return rows, cols, len(batch_indices), targets


class ScatterLoader:
    """Yields dense ``(data, target)`` batches built via on-device scatter.

    Drop-in replacement for ``DataLoader(CSRDataset(...))`` in the training loops: it
    exposes ``__iter__``, ``__len__`` and a ``.dataset`` attribute. Each batch is
    densified directly on ``device`` by scattering the one-hot column indices into a
    zeroed tensor, so only a handful of indices per position cross the CPU->GPU boundary
    instead of a full 772-wide dense row.
    """

    def __init__(self, dataset, batch_size=16, shuffle=True, device=None, drop_last=False):
        self.dataset = dataset
        self.device = device if device is not None else torch.device("cpu")
        base_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batch_sampler = BatchSampler(base_sampler, batch_size=batch_size, drop_last=drop_last)
        self.loader = DataLoader(dataset, sampler=batch_sampler, batch_size=None,
                                 collate_fn=_identity_collate)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        n_features = self.dataset.num_features
        for rows, cols, batch_size, targets in self.loader:
            data = torch.zeros(batch_size, n_features, device=self.device)
            data[rows.to(self.device), cols.to(self.device)] = 1.0
            yield data, targets.to(self.device)


def make_scatter_loader(features, results, batch_size=16, shuffle=True, device=None):
    """Build a :class:`ScatterLoader` over a CSR feature matrix and target array."""
    dataset = BatchIndexCSRDataset(features, results)
    return ScatterLoader(dataset, batch_size=batch_size, shuffle=shuffle, device=device)


def add_ocb(features):
    assert len(features.shape) == 2
    one_w_b = (features[:, (2*64):(3*64)].sum(axis=1) == 1)
    one_b_b = (features[:, (8*64):(9*64)].sum(axis=1) == 1)
    ds_b = 0
    for idx in [0, 2, 4, 6, 9, 11, 13, 15]:
        ds_b += features[:, (idx+2*64):(3*64):16].sum(axis=1) + features[:, (idx+8*64):(9*64):16].sum(axis=1)
    one_ds_b = ds_b == 1
    ocb = scipy.sparse.csr_matrix((one_w_b & one_b_b & one_ds_b).astype(np.int8))
    #print(f"W Bishop = {one_w_b}, B Bishop = {one_b_b}, DS Bishop = {one_ds_b}, OCB = {(one_w_b & one_b_b & one_ds_b).astype(np.int8)}")
    return scipy.sparse.hstack([features, ocb])


def load_dataset_ocb(name, batch_size=16, shuffle=True):
    features, results = load_features_results(name)
    features = add_ocb(features)
    return torch.utils.data.DataLoader(CSRDataset(features, results), batch_size=batch_size, shuffle=shuffle)


def load_from_multiple(lst, portion=1.0, save_dir="./"):
    """Load and concatenate several desk datasets, optionally subsampling each.

    Each element of ``lst`` is either a tag (e.g. ``"100a"``) loaded at the shared
    ``portion``, or a ``(tag, portion)`` tuple with a per-dataset fraction. When a
    portion < 1, a fresh random subset of that fraction is drawn each call -- so calling
    this repeatedly (see ``train_v2`` / ``train_net.py --reload-every``) streams different
    subsets over time while keeping only ~portion of the corpus resident at once.
    """
    def unpack(el):
        if isinstance(el, tuple):
            num, por = el
        else:
            num, por = el, portion
        return f"{save_dir}features_desk_v{num}.npz", f"{save_dir}targets_desk_v{num}.npz", por

    # Accumulate into lists and vstack/concatenate once at the end: vstack-ing inside the
    # loop recopies the whole growing matrix each iteration (O(n^2) peak memory and time).
    feature_blocks = []
    result_blocks = []
    for feature_filename, label_filename, por in (unpack(el) for el in lst):
        f0 = scipy.sparse.load_npz(feature_filename)
        r0 = np.load(label_filename)['arr_0']
        if por < 1.0:
            idx = np.arange(len(r0))
            random.shuffle(idx)
            m_idx = int(por * len(r0))
            keep = idx < m_idx
            f0 = f0[keep]
            r0 = r0[keep]
        feature_blocks.append(f0)
        result_blocks.append(r0)
    features = scipy.sparse.vstack(feature_blocks, format="csr")
    results = np.concatenate(result_blocks)
    return features, results
