import sys; sys.path.append('..')
import os
import re
import json
import random
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import dataset, DataLoader

from utils.logger import Logger


INPUT_DIR = Path.cwd().joinpath('../../input')
DATA_DIR = Path.cwd().joinpath('../../data')
SUBMIT_DIR = Path.cwd().joinpath('../../data')
LOG_DIR = Path.cwd().joinpath('../../logs')
SLACK_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['slack']

SEED = 42
PADDING_LENGTH = 60
EPOCHS = 8
THRESHOLD = 0.35


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


def preprocess_text(df):
    df['question_text'] = df['question_text'].apply(
        lambda x: replace_typical_misspell(
            clean_numbers(
                clean_text(
                    x.lower()
                )
            )
        )
    )
    return df


def tokenize_text(df, logger, tokenizer=None):
    text = df['question_text']

    if tokenizer is None:
        logger.info('Fit tokenizer on train data')
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(text))

    logger.info('Tokenize text data')
    sequence = tokenizer.texts_to_sequences(text)

    return sequence, tokenizer


def get_seq_length(sequence):
    return np.array([len(x) for x in sequence])


def load_embeddings(embed_type, word_index):
    if embed_type == 0:
        embedding_file = '../../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    elif embed_type == 1:
        embedding_file = '../../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    else:
        embedding_file = '../../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, encoding="utf8", errors='ignore')
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


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

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


def f1_from_logits(outputs: np.ndarray, targets: np.ndarray, threshold: float=0.5):
    outputs = outputs.reshape(-1,)
    targets = targets.reshape(-1,).astype(int)
    return f1_score(targets, sp.special.expit(outputs) > threshold)


def bce_from_logits(outputs, targets):
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    return bce_loss(outputs.detach(), targets.detach()).item()

# ================================== #


# ========== Data Loaders ========== #

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

# ================================== #


# ========== PyTorch Core ========== #

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def to_gpu(data, output_device):
    if isinstance(data, list):
        data = [x.to(output_device) for x in data]
    elif isinstance(data, dict):
        data = dict([(key, to_gpu(x, output_device)) for (key, x) in data.items()])
    else:
        data = data.to(output_device)
    return data


def to_cpu(data):
    if isinstance(data, list):
        data = [x.detach().cpu().numpy() for x in data]
    elif isinstance(data, dict):
        data = dict([(key, to_cpu(x)) for (key, x) in data.items()])
    else:
        data = data.detach().cpu().numpy()
    return data


def train_model(model, criteria, metric, optimizer, scheduler, dataloader, logger, config):
    epochs = config['epochs']
    loss_names = config['loss_names']
    metric_type = config['metric_type']

    for epoch in range(1, epochs + 1):
        logger.info(f'Epoch {epoch:d} / {epochs:d}')
        logger.info(f'Training phase starts.\n')
        model, train_losses, train_metric = train_on_epoch(model, criteria, metric, metric_type,
                                                           optimizer, dataloader, config)

        message = f'Training phase has been done. Epoch {epoch:d} / {epochs:d} results:\n'
        message += 'Train - ' + ', '.join([f'{name}: {loss:.5f}' for name, loss in zip(loss_names, train_losses)])
        message += f', metric: {train_metric:.5f}\n'
        logger.info(message)

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(metrics=train_metric)
            else:
                scheduler.step()

    return model


def train_on_epoch(model, criteria, metric, metric_type, optimizer, dataloader, config):
    output_device = config['output_device']
    reg_lambda = config['reg_lambda']

    train_losses = np.zeros(len(criteria[0]))
    train_metric = 0
    outputs_total = []
    targets_total = []
    n_iter = len(dataloader)
    model.train()

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = to_gpu(inputs, output_device)
        targets = to_gpu(targets, output_device)
        outputs = model(inputs)

        total_loss = 0
        for j, (criterion, coeff) in enumerate(zip(criteria[0], criteria[1])):
            loss = coeff * criterion(outputs, targets)
            total_loss += loss
            train_losses[j] += loss.item() / n_iter

        if reg_lambda is not None:
            l2_loss = L2Regulaization(reg_lambda)
            total_loss += l2_loss(model)

        if metric_type == 'cumulative':
            train_metric += metric(outputs, targets) / n_iter
        else:
            outputs_total.append(to_cpu(outputs))
            targets_total.append(to_cpu(targets))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step(closure=None)

    if metric_type != 'cumulative':
        outputs_total = np.concatenate(outputs_total, axis=0)
        targets_total = np.concatenate(targets_total, axis=0)
        train_metric = metric(outputs_total, targets_total)

    return model, train_losses, train_metric


def predict(model, dataloader, config):
    output_device = config['output_device']
    results = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            inputs = to_gpu(inputs, output_device)
            outputs = model.forward(inputs)
            outputs = to_cpu(outputs)
            results.append(outputs)

    if isinstance(results[0], np.ndarray):
        return np.concatenate(results, axis=0)
    elif isinstance(results[0], dict):
        return dict([(key, [d[key] for d in results]) for key in results[0].keys()])
    else:
        return results

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


class StackedRNNFM(nn.Module):
    def __init__(self, embedding_matrix, seq_len, hidden_size=64):
        super(StackedRNNFM, self).__init__()

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, seq_len)
        self.gru_attention = Attention(hidden_size * 2, seq_len)

        fm_first_size = hidden_size * 2 * 4
        fm_second_size = hidden_size * 2 * sp.special.comb(4, 2)

        self.fc = nn.Linear(int(fm_first_size + fm_second_size), 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, inputs):
        x_embedding = self.embedding(inputs)  # B x L x D
        x_embedding = self.embedding_dropout(torch.unsqueeze(x_embedding, 0).transpose(1, 3))
        x_embedding = torch.squeeze(x_embedding.transpose(1, 3))

        x_lstm, _ = self.lstm(x_embedding)
        x_gru, _ = self.gru(x_lstm)

        x_lstm_attention = self.lstm_attention(x_lstm)
        x_gru_attention = self.gru_attention(x_gru)
        x_avg_pool = torch.mean(x_gru, 1)
        x_max_pool, _ = torch.max(x_gru, 1)

        fm_first = [
            x_lstm_attention,
            x_gru_attention,
            x_avg_pool,
            x_max_pool
        ]

        fm_second = []
        for t_1, t_2 in itertools.combinations(fm_first, 2):
            fm_second.append(t_1 * t_2)

        x_fc = self.fc(torch.cat(fm_first + fm_second, 1))
        x_fc = self.dropout(self.relu(x_fc))
        outputs = self.output_layer(x_fc)
        return outputs

# ================================== #


def main(logger, args):
    with logger.timer('Data loading'):
        df_train, df_test = load_data(INPUT_DIR, logger)
    with logger.timer('Tokenize train data'):
        seq_train, tokenizer = tokenize_text(df_train, logger)
    with logger.timer('Tokenize test data'):
        seq_test, _ = tokenize_text(df_test, logger, tokenizer=tokenizer)
    with logger.timer('Pad train text data'):
        seq_train = pad_sequences(seq_train, maxlen=PADDING_LENGTH)
    with logger.timer('Pad test text data'):
        seq_test = pad_sequences(seq_test, maxlen=PADDING_LENGTH)

    label_train = df_train['target'].values.reshape(-1, 1)

    with logger.timer('Load embeddings'):
        embedding_matrix = load_embeddings(embed_type=0, word_index=tokenizer.word_index)

    # ===== training and evaluation loop ===== #

    with logger.timer('Training pre-settings'):
        device_ids = args['device_ids']
        output_device = device_ids[0]
        torch.cuda.set_device(device_ids[0])

        set_seed(SEED)

        batch_size = args['batch_size'] * len(device_ids)
        max_workers = args['max_workers']
        test_preds = np.zeros(seq_test.shape[0])

    with logger.timer('Dataloader preparation'):
        x_train, x_test = seq_train.astype(int), seq_test.astype(int)
        y_train = label_train.astype(np.float32)

        dataset_train = SimpleDataset(x_train, y_train)
        dataset_test = SimpleDataset(x_test)

        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

    with logger.timer('Build NN model'):
        model = StackedRNNFM(embedding_matrix, PADDING_LENGTH, hidden_size=64)
        model.to(output_device)

    with logger.timer('Model training preparation - loss and optimizer'):
        criteria = [
            [nn.BCEWithLogitsLoss(reduction='mean')], [1.0]
        ]
        metric = f1_from_logits
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None

        config = {
            'epochs': EPOCHS,
            'loss_names': ['BCE Loss'],
            'metric_type': 'batch',
            'output_device': output_device,
            'reg_lambda': None,
        }

    with logger.timer('Train model'):
        model = train_model(model, criteria, metric, optimizer, scheduler, dataloader_train, logger, config)

    with logger.timer('Predict test data'):
        test_preds += sp.special.expit(predict(model, dataloader_test, config).reshape(-1,))

    message = f'Training and prediction has been done.'
    logger.post(message)

    with logger.timer('Make submission'):
        submission = pd.read_csv(INPUT_DIR.joinpath('sample_submission.csv'))
        submission.prediction = test_preds > 0.33
        submission.to_csv(SUBMIT_DIR.joinpath('submission.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--device-ids', metavar='N', type=int, nargs='+', default=[0])
    parser.add_argument('--max-workers', default=2, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args().__dict__

    script_name = Path(__file__).stem
    log_dir = LOG_DIR.joinpath(f'{script_name}')
    logger = Logger(script_name, log_dir=log_dir, webhook_url=SLACK_URL, overwrite=True)
    logger.info(f'Script starts with command line arguments: {args}')
    try:
        main(logger, args)
        logger.post('===== Script completed successfully! =====')
    except Exception as e:
        logger.exception(e)
