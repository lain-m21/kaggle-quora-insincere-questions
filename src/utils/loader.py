from collections import defaultdict
import torch
from torch.utils.data import dataset, sampler


def collate_dict(inputs, index):
    if isinstance(inputs, dict):
        return dict([(key, collate_dict(item, index)) for key, item in inputs.items()])
    else:
        return inputs[index]


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


class BalancedSampler(sampler.Sampler):
    def __init__(self, data_source: dataset.Dataset, indices=None, num_samples=None):
        super(BalancedSampler, self).__init__(data_source)
        if indices is None:
            self.indices = list(range(len(data_source)))
        else:
            self.indices = indices

        if num_samples is None:
            self.num_samples = len(self.indices)
        else:
            self.num_samples = num_samples

        label_to_count = defaultdict(lambda: 0)
        for idx in self.indices:
            label = self._get_label(data_source, idx)
            label_to_count[label] += 1

        weights =[1.0 / label_to_count[self._get_label(data_source, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, data_source: dataset.Dataset, idx: int):
        _, label = data_source[idx]
        return label

    def __iter__(self):
        return [self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True)]

    def __len__(self):
        return self.num_samples

