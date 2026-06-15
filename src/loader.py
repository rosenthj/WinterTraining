import os

import numpy as np
import scipy
import torch
import random

from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler


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
    def unpack(el):
        if isinstance(el, tuple):
            num, por = el
        else:
            num, por = el, portion
        return f"{save_dir}features_desk_v{num}.npz", f"{save_dir}targets_desk_v{num}.npz", por

    lst = [unpack(element) for element in lst]
    # lst = [(f"{save_dir}features_desk_v{num}.npz", f"{save_dir}targets_desk_v{num}.npz") for num in lst]
    f, r = None, None
    for feature_filename, label_filename, portion in lst:
        f0 = scipy.sparse.load_npz(feature_filename)
        r0 = np.load(label_filename)['arr_0']
        if portion < 1.0:
            idx = np.arange(len(r0))
            random.shuffle(idx)
            m_idx = int(portion * len(r0))
            f0 = f0[idx < m_idx]
            r0 = r0[idx < m_idx]
        if f is None:
            f = f0
            r = r0
        else:
            f = scipy.sparse.vstack([f, f0])
            r = np.concatenate([r, r0])
    return f, r
