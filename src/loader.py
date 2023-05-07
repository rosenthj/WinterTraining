import numpy as np
import scipy
import torch


def load_features_results(name):
    features = scipy.sparse.load_npz(f"{name}_features.npz")
    results = np.load(f"{name}_targets.npz")['arr_0']
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


def add_ocb(features):
    assert len(features.shape) == 2
    one_w_b = (features[:, (2*64):(3*64)].sum(axis=1) == 1)
    one_b_b = (features[:, (8*64):(9*64)].sum(axis=1) == 1)
    ds_b = 0
    for idx in [0,2,4,6,9,11,13,15]:
        ds_b += features[:, (idx+2*64):(3*64):16].sum(axis=1) + features[:, (idx+8*64):(9*64):16].sum(axis=1)
    one_ds_b = ds_b == 1
    ocb = scipy.sparse.csr_matrix((one_w_b & one_b_b & one_ds_b).astype(np.int8))
    #print(f"W Bishop = {one_w_b}, B Bishop = {one_b_b}, DS Bishop = {one_ds_b}, OCB = {(one_w_b & one_b_b & one_ds_b).astype(np.int8)}")
    return scipy.sparse.hstack([features, ocb])


def load_dataset_ocb(name, batch_size=16, shuffle=True):
    features, results = load_features_results(name)
    features = add_ocb(features)
    return torch.utils.data.DataLoader(CSRDataset(features,results), batch_size=batch_size, shuffle=shuffle)


def load_from_multiple(lst, save_dir="./"):
    lst = [(f"{save_dir}features_desk_v{num}.npz", f"{save_dir}targets_desk_v{num}.npz") for num in lst]
    f, r = None, None
    for feature_filename, label_filename in lst:
        if f is None:
            f = add_ocb(scipy.sparse.load_npz(feature_filename))
            r = np.load(label_filename)['arr_0']
        else:
            f = scipy.sparse.vstack([f, add_ocb(scipy.sparse.load_npz(feature_filename))])
            r = np.concatenate([r, np.load(label_filename)['arr_0']])
    return f, r