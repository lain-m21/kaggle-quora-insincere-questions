import sys; sys.path.append('..')
import os
import re
import time
import itertools
from contextlib import contextmanager
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset, DataLoader, sampler


INPUT_DIR = Path.cwd().joinpath('../input')
DATA_DIR = Path.cwd().joinpath('../../data')
SUBMIT_DIR = Path.cwd()

SEED = 42
PADDING_LENGTH = 70
EPOCHS = 5
TRIGGER = 1
NUM_SNAPSHOTS = 5
NUM_SEED_AVG = 3
THRESHOLDS = [0.365, 0.377, 0.379, 0.377, 0.37]
MODE = 'majority'
GLOBAL_THRESHOLD = 0.37

args = {
    'batch_size': 512,
    'device_ids': [0],
    'max_workers': 2
}


class SimpleLogger:
    def info(self, msg):
        print(msg)

    def post(self, msg):
        self.info(msg)

    @contextmanager
    def timer(self, process_name: str):
        since = time.time()
        yield
        message = f'Process [{process_name}] finished in {time.time() - since:.2f} sec'
        self.info(message)


# ======== Data Preparation ======== #

def load_data(input_dir, logger):
    logger.info('Loading train and test data from {}'.format(input_dir))
    df_train = pd.read_csv(input_dir.joinpath('train.csv').open('r'))
    df_test = pd.read_csv(input_dir.joinpath('test.csv').open('r'))
    logger.info('Train shape: {}'.format(df_train.shape))
    logger.info('Test shape: {}'.format(df_test.shape))
    return df_train, df_test


puncts = [
    ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+',
    '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×',
    '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±',
    '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄',
    '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，',
    '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√',
]

mispell_dict = {
    "aren't": "are not", "can't": "cannot", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'll": "he will", "he's": "he is", "i'd": "I would", "i'll": "I will", "i'm": "I am", "isn't": "is not",
    "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us", "mightn't": "might not",
    "mustn't": "must not", "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
    "shouldn't": "should not", "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have", "we'd": "we would", "we're": "we are",
    "weren't": "were not", "we've": "we have", "what'll": "what will", "what're": "what are", "what's": "what is",
    "what've": "what have", "where's": "where is", "who'd": "who would", "who'll": "who will", "who're": "who are",
    "who's": "who is", "who've": "who have", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have", "'re": " are", "wasn't": "was not",
    "we'll": " will", "tryin'": "trying"
}


def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub(r'[0-9]{5,}', '#####', x)
    x = re.sub(r'[0-9]{4}', '####', x)
    x = re.sub(r'[0-9]{3}', '###', x)
    x = re.sub(r'[0-9]{2}', '##', x)
    return x


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispellings, mispellings_re = _get_mispell(mispell_dict)


def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


def preprocess_text(df, return_df=False):
    processed_texts = df['question_text'].apply(
        lambda x: replace_typical_misspell(
            clean_numbers(
                clean_text(
                    x.lower()
                )
            )
        )
    )
    if return_df:
        df['question_text'] = processed_texts
        return df
    else:
        return processed_texts


def tokenize_texts(texts, logger, tokenizer=None, texts_another=None):
    if tokenizer is None:
        if texts_another is None:
            logger.info('Fit tokenizer on train data')
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(list(texts))
        else:
            logger.info('Fit tokenizer on train and test data')
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(list(texts) + list(texts_another))

    logger.info('Tokenize text data')
    sequence = tokenizer.texts_to_sequences(texts)

    return sequence, tokenizer


def extract_nlp_features(df):
    df['total_length'] = df['question_text'].apply(len)
    df['n_capitals'] = df['question_text'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['n_words'] = df['question_text'].str.count('\S+')
    df['n_unique_words'] = df['question_text'].apply(lambda x: len(set(w for w in x.split())))
    df['n_puncts'] = df['question_text'].apply(lambda x: sum(1 for c in x if c in set(puncts)))
    df['n_?'] = df['question_text'].apply(lambda x: sum(1 for c in x if c == '?'))
    df['n_!'] = df['question_text'].apply(lambda x: sum(1 for c in x if c == '!'))
    df['n_you'] = df['question_text'].apply(lambda x: len(re.findall(r'you[^~a-z]', x.lower())))
    return df


def load_embeddings(embed_type, word_index):
    if embed_type == 0:
        embedding_file = INPUT_DIR.joinpath('embeddings/glove.840B.300d/glove.840B.300d.txt')
    elif embed_type == 1:
        embedding_file = INPUT_DIR.joinpath('embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
    else:
        embedding_file = INPUT_DIR.joinpath('embeddings/paragram_300_sl999/paragram_300_sl999.txt')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in embedding_file.open(encoding='utf8', errors='ignore')
                            if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = len(word_index) + 1
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)).astype(np.float32)
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_multiple_embeddings(word_index, embed_types=(0, 1), max_workers=2):
    results = []
    for e_type in embed_types:
        results.append(load_embeddings(e_type, word_index))
    return results


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def parallel_apply(func_and_args):
    func, func_args = func_and_args
    return func(*func_args)

# ================================== #


# ======== Loss and Metrics ======== #

class L2Regulaization(nn.Module):
    exclude_regex = re.compile(r'bn|bias|activation')

    def __init__(self, reg_lambda):
        super(L2Regulaization, self).__init__()
        self.reg_lambda = reg_lambda

    def forward(self, model: nn.Module):
        loss = 0
        for name, weights in model.named_parameters():
            if self.exclude_regex.search(name):
                continue
            loss += weights.norm(2)

        return loss * self.reg_lambda

# ================================== #


# ========== Data Loaders ========== #

def collate_dict(inputs, index):
    if isinstance(inputs, dict):
        return dict([(key, collate_dict(item, index)) for key, item in inputs.items()])
    elif isinstance(inputs, list):
        return [collate_dict(item, index) for item in inputs]
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
        pos_count = int(self.pos_ratio * self.num_samples)
        neg_count = self.num_samples - pos_count
        samples_positive = np.random.choice(self.indices_positive, pos_count)
        samples_negative = np.random.choice(self.indices_negative, neg_count)
        return np.concatenate([samples_positive, samples_negative])

    def _get_indices(self, samples):
        if self.shuffle:
            np.random.shuffle(samples)
        return samples

    def __iter__(self):
        return iter(self._get_indices(self._get_samples()))

    def __len__(self):
        return self.num_samples


class BinaryOverSampler(sampler.Sampler):
    def __init__(self, labels, over_sample_factor=2, shuffle=True):
        super(BinaryOverSampler, self).__init__(labels)

        self.indices_positive = np.where(labels == 1)[0]
        self.indices_negative = np.where(labels == 0)[0]
        self.over_sample_factor = over_sample_factor
        self.shuffle = shuffle
        self.num_samples = len(self.indices_negative) + int(len(self.indices_positive * over_sample_factor))

    def _get_samples(self):
        pos_count = int(self.over_sample_factor * len(self.indices_positive))
        samples_positive = np.random.choice(self.indices_positive, pos_count)
        samples_negative = self.indices_negative
        return np.concatenate([samples_positive, samples_negative])

    def _get_indices(self, samples):
        if self.shuffle:
            np.random.shuffle(samples)
        return samples

    def __iter__(self):
        return iter(self._get_indices(self._get_samples()))

    def __len__(self):
        return self.num_samples

# ================================== #


# ========== PyTorch Core ========== #

def set_torch_environment(num_threads=2):
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, model, logger, config):
        self.model = model
        self.logger = logger

        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.num_snapshots = config['num_snapshots']

        # transfer model to GPU
        self.output_device = config['output_device']
        self.model.to(self.output_device)

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=config['optimizer_lr'])
        elif config['optimizer'] == 'asdg':
            self.optimizer = optim.ASGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=config['optimizer_lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'sparse_adam':
            self.optimizer = optim.SparseAdam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=config['optimizer_lr'])
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                       lr=config['optimizer_lr'], momentum=config['momentum'],
                                       weight_decay=config['weight_decay'], nesterov=config['nesterov'])
        elif config['optimizer'] == 'rmsprop':
            self.optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=config['optimizer_lr'], weight_decay=config['weight_decay'],
                                           momentum=config['momentum'])
        else:
            raise ValueError('The selected optimizer is not supported for this trainer.')

        # scheduler
        self.scheduler_type = config['scheduler_type']

        if self.scheduler_type == 'cyclic':
            self.scheduler = CyclicLRScheduler(self.optimizer,
                                               base_lr=config['base_lr'],
                                               max_lr=config['max_lr'],
                                               step_size=config['step_size'],
                                               mode=config['scheduler_mode'],
                                               gamma=config['scheduler_gamma'])
            self.scheduler_trigger_steps = config['scheduler_trigger_steps']
        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=config['scheduler_milestones'],
                                                            gamma=config['scheduler_gamma'])
        else:
            self.scheduler = None

        # data sampler
        self.sampler_type = config['sampler_type']
        if self.sampler_type == 'over':
            self.over_sample_factor = config['over_sample_factor']
        elif self.sampler_type == 'balanced':
            self.pos_ratio = config['pos_ratio']
            self.num_samples = config['num_samples']

        self._set_torch_seed(config['seed'])

    def train_and_predict(self, x_train, y_train, x_test, thresholds):
        train_loader = self.get_loader(x_train, y_train, mode='train')
        test_loader = self.get_loader(x_test, mode='test')

        n_iter = len(train_loader)
        checkpoint_count = 0
        step_count = 0

        predict_results = []
        for epoch in range(self.epochs):
            self.model.train()
            loss_epoch = 0
            self.logger.info(f'Epoch: {epoch + 1} / {self.epochs}')
            for inputs, targets in train_loader:
                inputs, targets = self._tensors_to_gpu(inputs), self._tensors_to_gpu(targets)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss_epoch += loss.item() / n_iter

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(closure=None)
                if self.scheduler_type == 'cyclic' and step_count > self.scheduler_trigger_steps:
                    checkpoint_flag = self.scheduler.batch_step()
                    if checkpoint_flag:
                        self.logger.info('Cyclic scheduler hit the bottom. Start prediction.')
                        predict_results.append(self.predict(test_loader, thresholds[checkpoint_count]))
                        checkpoint_count += 1
                        self.logger.info('Prediction has finished.')

                        if checkpoint_count >= self.num_snapshots:
                            break
                        self.model.train()

                step_count += 1

            if self.scheduler_type == 'step':
                self.scheduler.step()

        if self.scheduler_type != 'cyclic':
            predict_results.append(self.predict(test_loader, thresholds[0]))

        return predict_results

    def predict(self, test_loader, threshold):
        total_outputs = []

        self.model.eval()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = self._tensors_to_gpu(inputs)
                outputs = self.model(inputs)
                total_outputs.append(self._tensors_to_numpy(outputs))

        total_outputs = np.concatenate(total_outputs, axis=0)
        preds = sp.special.expit(total_outputs)
        predict_result = {
            'preds_proba': preds.reshape(-1,),
            'preds_binary': np.array(preds > threshold, dtype=int).reshape(-1,)
        }
        return predict_result

    def get_loader(self, x, y=None, mode='train'):
        if isinstance(x, dict):
            dataset = DictDataset(x, y)
        else:
            dataset = SimpleDataset(x, y)

        if mode == 'train':
            if self.sampler_type == 'over':
                sampler = BinaryOverSampler(y, self.over_sample_factor, shuffle=True)
            elif self.sampler_type == 'balanced':
                sampler = BinaryBalancedSampler(y, self.pos_ratio, self.num_samples, shuffle=True)
            else:
                sampler = None

            if sampler is not None:
                dataloader = DataLoader(dataset,
                                        batch_size=self.batch_size,
                                        sampler=sampler,
                                        pin_memory=True,
                                        worker_init_fn=worker_init_fn)
            else:
                dataloader = DataLoader(dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=worker_init_fn)
        else:
            dataloader = DataLoader(dataset,
                                    batch_size=self.batch_size * 4,
                                    pin_memory=True)

        return dataloader

    def _tensors_to_gpu(self, tensors):
        if isinstance(tensors, list):
            tensors = [x.to(self.output_device) for x in tensors]
        elif isinstance(tensors, dict):
            tensors = dict([(key, self._tensors_to_gpu(x)) for (key, x) in tensors.items()])
        else:
            tensors = tensors.to(self.output_device)
        return tensors

    def _tensors_to_numpy(self, tensors):
        if isinstance(tensors, list):
            tensors = [x.detach().cpu().numpy() for x in tensors]
        elif isinstance(tensors, dict):
            tensors = dict([(key, self._tensors_to_numpy(x)) for (key, x) in tensors.items()])
        else:
            tensors = tensors.detach().cpu().numpy()
        return tensors

    @staticmethod
    def _set_torch_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


class CyclicLRScheduler:
    def __init__(self, optimizer, base_lr=1e-5, max_lr=3e-3,
                 step_size=1200, mode='triangular', gamma=0.9,
                 last_batch_iteration=-1):
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

        self.restart = False
        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1

        # restart - lr jump up to the max_lr
        if self.restart:
            batch_iteration = 0
            self.restart = False

        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        # restart flag
        if (batch_iteration + 1) % self.step_size == 0:
            self.restart = True
            return True  # model checkpoint flag for snapshot
        else:
            return False

    def get_lr(self):
        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = max_lr - base_lr
            lr = base_lr + base_height * self._get_scaling()
            lrs.append(lr)
        return lrs

    def _get_scaling(self):
        if self.mode == 'triangular':
            return 1 - self.last_batch_iteration / self.step_size
        elif self.mode == 'cosine':
            return (1 + np.cos(np.pi * self.last_batch_iteration / self.step_size)) / 2
        else:
            return 1 - self.last_batch_iteration / self.step_size

# ================================== #

# =========== NN Modules =========== #


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, masking=True):
        super(Attention, self).__init__()

        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.supports_masking = masking
        self.bias = bias

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, inputs, mask=None):
        eij = torch.mm(
            inputs.contiguous().view(-1, self.feature_dim),
            self.weight
        ).view(-1, self.step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_inputs = inputs * torch.unsqueeze(a, -1)
        return torch.sum(weighted_inputs, 1)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


class Dense(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, activation='relu'):
        super(Dense, self).__init__()

        self.fc = nn.Linear(in_features, out_features)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.fc(inputs)
        x = self.activation(x)
        outputs = self.dropout(x)
        return outputs


class NLPFeaturesRNN(nn.Module):
    def __init__(self, input_shapes, embedding_matrix, seq_len, hidden_size=64, out_hidden_dim=64,
                 nlp_hidden_dim=64, nlp_dropout=0.2, embed_drop=0.2, out_drop=0.3, mask=False,
                 num_dense_layers=1, dense_activate='relu', embed_drop_direction=0, factorize=False):
        super(NLPFeaturesRNN, self).__init__()

        self.mask = mask
        self.factorize = factorize

        nlp_layers = []
        for i in range(num_dense_layers):
            if i == 0:
                layers = nn.ModuleList([
                    Dense(1, nlp_hidden_dim, dropout=nlp_dropout, activation=dense_activate)
                    for _ in range(input_shapes['nlp'])
                ])
            else:
                layers = nn.ModuleList([
                    Dense(nlp_hidden_dim, nlp_hidden_dim, dropout=nlp_dropout, activation=dense_activate)
                    for _ in range(input_shapes['nlp'])
                ])
            nlp_layers.append(layers)

        self.nlp_layers = nn.ModuleList(nlp_layers)

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(embed_drop)
        self.embed_drop_direction = embed_drop_direction

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, seq_len)
        self.gru_attention = Attention(hidden_size * 2, seq_len)

        fm_first_size_nlp = nlp_hidden_dim * input_shapes['nlp']
        fm_second_size_nlp = nlp_hidden_dim * sp.special.comb(input_shapes['nlp'], 2)

        fm_first_size_seq = hidden_size * 2 * 4
        fm_second_size_seq = hidden_size * 2 * sp.special.comb(4, 2)

        if factorize:
            fc_size = fm_first_size_seq + fm_second_size_seq + fm_first_size_nlp + fm_second_size_nlp
        else:
            fc_size = fm_first_size_seq + fm_second_size_seq + fm_first_size_nlp

        self.fc = nn.Linear(int(fc_size), out_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_drop)
        self.output_layer = nn.Linear(out_hidden_dim, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs['text'])  # B x L x D
        if self.embed_drop_direction == 0:
            x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
            x_embedding = torch.squeeze(x_embedding.transpose(1, 3))
        else:
            x_embedding = self.embedding_dropout(x_embedding)

        x_lstm, _ = self.lstm(x_embedding)
        x_gru, _ = self.gru(x_lstm)

        if self.mask:
            mask = get_non_pad_mask(inputs['text'])
            x_lstm_attention = self.lstm_attention(x_lstm, mask.squeeze(-1))
            x_gru_attention = self.gru_attention(x_gru, mask.squeeze(-1))
            x_avg_pool = torch.mean(x_gru * mask, 1)
            x_max_pool, _ = torch.max(x_gru * mask, 1)
        else:
            x_lstm_attention = self.lstm_attention(x_lstm)
            x_gru_attention = self.gru_attention(x_gru)
            x_avg_pool = torch.mean(x_gru, 1)
            x_max_pool, _ = torch.max(x_gru, 1)

        x_nlp = []
        for i, layers in enumerate(self.nlp_layers):
            if i == 0:
                nlp_inputs = inputs['nlp']
            else:
                nlp_inputs = x_nlp
                x_nlp = []
            for x, layer in zip(nlp_inputs, layers):
                x_nlp.append(layer(x))

        x_nlp_second = []
        if self.factorize:
            for t_1, t_2 in itertools.combinations(x_nlp, 2):
                x_nlp_second.append(t_1 * t_2)

        fm_first = [
            x_lstm_attention,
            x_gru_attention,
            x_avg_pool,
            x_max_pool
        ]

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second + x_nlp + x_nlp_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs

# ================================== #


def train_nlprnn(seed, seq_train, seq_test, nlp_train, nlp_test, label_train,
                 thresholds, mode, config, embedding_matrix, logger):
    np.random.seed(seed)
    logger.info(f'Seed average: seed = {seed}')
    with logger.timer('Dataloader preparation'):
        x_train = {
            'text': seq_train.astype(int),
            'nlp': nlp_train
        }
        x_test = {
            'text': seq_test.astype(int),
            'nlp': nlp_test
        }
        y_train = label_train.astype(np.float32)

    with logger.timer('Build NN model'):
        model = NLPFeaturesRNN({'nlp': len(nlp_train)}, embedding_matrix, PADDING_LENGTH,
                               hidden_size=64, out_hidden_dim=64, out_drop=0.3, embed_drop=0.2,
                               dense_activate='relu', nlp_hidden_dim=16, mask=True,
                               nlp_dropout=0.2, factorize=False,
                               num_dense_layers=2)

    with logger.timer('Trainer setup'):
        trainer = Trainer(model, logger, config)

    with logger.timer('Train model and make prediction'):
        predict_results = trainer.train_and_predict(x_train, y_train, x_test, thresholds)

    if mode == 'majority':
        test_preds = np.array([res['preds_binary'] for res in predict_results])
    else:
        test_preds = np.array([res['preds_proba'] for res in predict_results])

    return test_preds

# ================================== #


def main(logger, args):
    with logger.timer('Data loading'):
        df_train, df_test = load_data(INPUT_DIR, logger)
    with logger.timer('Preprocess train data'):
        df_train = preprocess_text(df_train)
    with logger.timer('Preprocess test data'):
        df_test = preprocess_text(df_test)
    with logger.timer('Tokenize train data'):
        seq_train, tokenizer = tokenize_texts(df_train['question_text'], logger, texts_another=df_test['question_text'])
    with logger.timer('Tokenize test data'):
        seq_test, _ = tokenize_texts(df_test['question_text'], logger, tokenizer=tokenizer)
    with logger.timer('Pad train text data'):
        seq_train = pad_sequences(seq_train, maxlen=PADDING_LENGTH)
    with logger.timer('Pad test text data'):
        seq_test = pad_sequences(seq_test, maxlen=PADDING_LENGTH)

    label_train = df_train['target'].values.reshape(-1, 1)

    embed_types = [0, 1]

    df_concat = pd.concat([df_train, df_test])

    logger.info('Start multiprocess nlp feature extraction and embedding matrices loading')
    with logger.timer('Parallel nlp feature extraction and embedding matrices loading'), mp.Pool(processes=2) as p:
        results = p.map(parallel_apply, [
            (extract_nlp_features, (df_concat, )),
            (load_multiple_embeddings, (tokenizer.word_index, embed_types, args['debug']))
        ])

    df_extracted = results[0]
    embedding_matrices = results[1]
    embedding_matrix = np.concatenate(embedding_matrices, axis=1)

    nlp_columns = ['total_length', 'n_capitals', 'n_words', 'n_puncts', 'n_?', 'n_!', 'n_you']
    for col in nlp_columns:
        scaler = StandardScaler()
        df_extracted[col] = scaler.fit_transform(
            df_extracted[col].values.astype(np.float32).reshape(-1, 1)).reshape(-1, )

    x_nlp_train = [df_extracted[col].values[:len(df_train)].reshape(-1, 1) for col in nlp_columns]
    x_nlp_test = [df_extracted[col].values[len(df_train):].reshape(-1, 1) for col in nlp_columns]

    # ===== training and evaluation loop ===== #

    with logger.timer('Training pre-settings'):
        device_ids = args['device_ids']
        output_device = device_ids[0]
        torch.cuda.set_device(device_ids[0])

        batch_size = args['batch_size'] * len(device_ids)
        trigger = TRIGGER

        steps_per_epoch = seq_train.shape[0] // batch_size
        scheduler_trigger_steps = steps_per_epoch * trigger
        step_size = steps_per_epoch * (EPOCHS - trigger) // NUM_SNAPSHOTS

    test_preds_total = []

    for i in range(NUM_SEED_AVG):
        logger.info(f'Start train and prediction {i + 1} / {NUM_SEED_AVG}')
        config = {
            'epochs': EPOCHS,
            'batch_size': batch_size,
            'output_device': output_device,
            'optimizer': 'adam',
            'optimizer_lr': 0.003,
            'num_snapshots': NUM_SNAPSHOTS,
            'scheduler_type': 'cyclic',
            'base_lr': 0.0005,
            'max_lr': 0.003,
            'step_size': step_size,
            'scheduler_mode': 'triangular',
            'scheduler_gamma': 0.9,
            'scheduler_trigger_steps': scheduler_trigger_steps,
            'sampler_type': 'normal',
            'seed': SEED * i
        }
        test_preds = train_nlprnn(
            seed=SEED * i,
            seq_train=seq_train,
            seq_test=seq_test,
            nlp_train=x_nlp_train,
            nlp_test=x_nlp_test,
            label_train=label_train,
            thresholds=THRESHOLDS,
            mode=MODE,
            config=config,
            embedding_matrix=embedding_matrix,
            logger=logger
        )
        test_preds_total.append(test_preds)

    message = f'Training and prediction has been done.'
    logger.post(message)
    if MODE == 'majority':
        test_preds_total = np.concatenate(test_preds_total, axis=0).mean(0) > 0.5
    else:
        test_preds_total = np.concatenate(test_preds_total, axis=0).mean(0) > GLOBAL_THRESHOLD

    with logger.timer('Make submission'):
        submission = pd.read_csv(INPUT_DIR.joinpath('sample_submission.csv'))
        submission.prediction = test_preds_total
        submission.to_csv(SUBMIT_DIR.joinpath('submission.csv'), index=False)


if __name__ == '__main__':
    logger = SimpleLogger()
    main(logger, args)
