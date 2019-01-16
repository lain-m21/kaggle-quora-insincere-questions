from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import dataset, sampler


def collate_dict(inputs, index):
    if isinstance(inputs, dict):
        return dict([(key, collate_dict(item, index)) for key, item in inputs.items()])
    else:
        return inputs[index]


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class SimpleDataset(dataset.Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        X = self.X[idx]
        if self.y is not None:
            y = self.y[idx]
        else:
            y = 0
        return X, y

    def __len__(self):
        return self.X.shape[0]


class DictDataset(dataset.Dataset):
    def __init__(self, inputs, outputs=None, data_size=None):
        self.inputs = inputs
        self.outputs = outputs
        self.data_size = data_size

    def __getitem__(self, idx):
        inputs = collate_dict(self.inputs, idx)
        if self.outputs is not None:
            if isinstance(self.outputs, dict):
                outputs = collate_dict(self.outputs, idx)
            else:
                outputs = self.outputs[idx]
        else:
            outputs = 0
        return inputs, outputs

    def __len__(self):
        if self.data_size is not None:
            return self.data_size
        else:
            key = list(self.inputs.keys())[0]
            return self.inputs[key].shape[0]


class BinaryBalancedSampler(sampler.Sampler):
    def __init__(self, labels, pos_ratio=0.5, num_samples=None, shuffle=True):
        super(BinaryBalancedSampler, self).__init__(labels)

        self.indices_positive = np.where(labels == 1)[0]
        self.indices_negative = np.where(labels == 0)[0]
        self.pos_ratio = pos_ratio
        if num_samples is None:
            self.num_samples = len(labels)
        else:
            self.num_samples = num_samples
        self.shuffle = shuffle

    def _get_samples(self):
        pos_count = self.pos_ratio * self.num_samples
        neg_count = self.num_samples - pos_count
        if self.pos_ratio < 0.5:
            pos_replace = True
            neg_replace = False
        else:
            pos_replace = False
            neg_replace = True
        samples_positive = np.random.choice(self.indices_positive, pos_count, replace=pos_replace)
        samples_negative = np.random.choice(self.indices_negative, neg_count, replace=neg_replace)
        return np.concatenate([samples_positive, samples_negative])

    def _get_indices(self, samples):
        if self.shuffle:
            np.random.shuffle(samples)
        return samples

    def __iter__(self):
        return iter(self._get_indices(self._get_samples()))

    def __len__(self):
        return self.num_samples

