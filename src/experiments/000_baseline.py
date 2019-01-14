import sys; sys.path.append('..')
import json
import argparse
from pathlib import Path
import numpy as np
import scipy as sp
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.logger import Logger
from utils.prepare_data import load_data, tokenize_text, load_embeddings
from utils.pytorch_core import train_model, predict, set_seed
from utils.loader import SimpleDataset

from utils.metrics import f1_from_logits, threshold_search
from model.baseline_rnn import StackedRNNFM


INPUT_DIR = Path.cwd().joinpath('../../input')
DATA_DIR = Path.cwd().joinpath('../../data')
LOG_DIR = Path.cwd().joinpath('../../logs')
SLACK_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['slack']

SEED = 42
PADDING_LENGTH = 60
EPOCHS = 8
KFOLD = 5


def main(logger, args):
    df_train, df_test = load_data(INPUT_DIR, logger)
    seq_train, tokenizer = tokenize_text(df_train, logger)

    logger.info('Pad train text data')
    seq_train = pad_sequences(seq_train, maxlen=PADDING_LENGTH)

    label_train = df_train['target'].values.reshape(-1, 1)

    if args['debug']:
        embedding_matrix = np.random.rand(len(tokenizer.word_index), 300)
    else:
        embedding_matrix = load_embeddings(tokenizer.word_index, logger, embed_type=0)

    # ===== training and evaluation loop ===== #
    device_ids = args['device_ids']
    output_device = device_ids[0]
    torch.cuda.set_device(device_ids[0])

    set_seed(SEED)

    batch_size = args['batch_size'] * len(device_ids)
    max_workers = args['max_workers']
    if args['debug']:
        epochs = 2
    else:
        epochs = EPOCHS

    logger.info('Start training and evaluation loop')

    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(seq_train.shape[0])
    for fold, (index_train, index_valid) in enumerate(skf.split(label_train, label_train)):
        x_train, x_valid = seq_train[index_train].astype(int), seq_train[index_valid].astype(int)
        y_train, y_valid = label_train[index_train].astype(np.float32), label_train[index_valid].astype(np.float32)

        dataset_train = SimpleDataset(x_train, y_train)
        dataset_valid = SimpleDataset(x_valid, y_valid)

        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        dataloader_valid = DataLoader(
            dataset=dataset_valid,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
        dataloaders = {'train': dataloader_train, 'valid': dataloader_valid}

        model = StackedRNNFM(embedding_matrix, PADDING_LENGTH, hidden_size=32)
        model.to(output_device)

        criteria = [
            [nn.BCEWithLogitsLoss(reduction='mean')], [1.0]
        ]
        metric = f1_from_logits
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None

        model_save_path = str(DATA_DIR.joinpath(f'models/{Path(__file__).stem}_fold_{fold}.model'))
        model_name = model._get_name()
        config = {
            'epochs': epochs,
            'loss_names': ['BCE Loss'],
            'metric_type': 'batch',
            'model_save_path': model_save_path,
            'output_device': output_device,
            'mode': 'min',
            'early_stopping': 200,
            'model_name': model_name,
            'reg_lambda': None,
            'fold': fold
        }

        model, valid_score = train_model(model, criteria, metric, optimizer, scheduler, dataloaders, logger, config)

        message = f'Training and evaluation for the fold {fold + 1} / {KFOLD} has been done.\n'
        message += f'Validation F1 score: {valid_score}\n'
        logger.post(message)

        dataloader_valid = DataLoader(
            dataset=dataset_valid,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
        oof_preds[index_valid] = sp.special.expit(predict(model, dataloader_valid, config).reshape(-1,))

    logger.info('Training and evaluation loop has been done. Start f1 threshold search.')
    search_result = threshold_search(label_train.reshape(-1,), oof_preds)
    logger.post(f'Threshold search result - f1: {search_result["f1"]}, threshold: {search_result["threshold"]}')


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
