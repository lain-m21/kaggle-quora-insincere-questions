import sys; sys.path.append('..')
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from keras.preprocessing.sequence import pad_sequences

import torch

from utils.logger import Logger, post_to_main_spreadsheet, post_to_snapshot_spreadsheet
from utils.prepare_data import load_data, tokenize_text, load_multiple_embeddings, preprocess_text
from utils.trainer import Trainer

from model.rnn import AttentionMaskRNNAnother
from model.cnnrnn import BranchedMaskCNNRNN, BranchedMaskCNNRNNAnother


INPUT_DIR = Path.cwd().joinpath('../../input')
DATA_DIR = Path.cwd().joinpath('../../data')
LOG_DIR = Path.cwd().joinpath('../../logs')
SLACK_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['slack']
SPREADSHEET_MAIN_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['spreadsheet_main']
SPREADSHEET_SNAPSHOT_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['spreadsheet_snapshots']

SCRIPT_NAME = Path(__file__).stem

SEED = 42
PADDING_LENGTH = 60
EPOCHS = 5
NUM_SNAPSHOTS = 5
KFOLD = 5


def main(logger, args):
    df_train, _ = load_data(INPUT_DIR, logger)
    logger.info('Preprocess text')
    if args['debug']:
        df_train = df_train.iloc[:200000]
    else:
        df_train = preprocess_text(df_train)
    seq_train, tokenizer = tokenize_text(df_train, logger)

    logger.info('Pad train text data')
    seq_train = pad_sequences(seq_train, maxlen=PADDING_LENGTH)

    label_train = df_train['target'].values.reshape(-1, 1)

    if args['debug']:
        embedding_matrix = np.random.rand(len(tokenizer.word_index) + 1, 300).astype(np.float32)
    else:
        logger.info('Load multiple embeddings')
        embedding_matrices = load_multiple_embeddings(tokenizer.word_index, embed_types=[0, 2],
                                                      max_workers=args['max_workers'])
        embedding_matrix = np.array(embedding_matrices).mean(0)

    # ===== training and evaluation loop ===== #
    device_ids = args['device_ids']
    output_device = device_ids[0]
    torch.cuda.set_device(device_ids[0])
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    batch_size = args['batch_size'] * len(device_ids)
    epochs = EPOCHS

    logger.info('Start training and evaluation loop')

    model_specs = [
        {'architecture': 'cnn', 'mask': True, 'pool_type': 'avg'},
        {'architecture': 'cnn', 'mask': True, 'pool_type': 'max'},
        {'architecture': 'cnn', 'mask': True, 'pool_type': 'both'},
        {'architecture': 'cnn', 'mask': False, 'pool_type': 'both'},
        {'architecture': 'cnn_another', 'mask': True, 'pool_type': 'avg'},
        {'architecture': 'cnn_another', 'mask': True, 'pool_type': 'max'},
        {'architecture': 'cnn_another', 'mask': True, 'pool_type': 'both'},
        {'architecture': 'cnn_another', 'mask': False, 'pool_type': 'both'},
        {'architecture': 'rnn', 'mask': True},
        {'architecture': 'rnn', 'mask': False}
    ]

    model_name_base_rnn = 'AttentionMaskRNNAnother'
    model_name_base_cnn = 'BranchedMaskCNNRNN'
    model_name_base_cnn_another = 'BranchedMaskCNNRNNAnother'

    for spec_id, spec in enumerate(model_specs):
        if spec['architecture'] == 'rnn':
            model_name = model_name_base_rnn + f'_specId={spec_id}_mask={spec["mask"]}'
        elif spec['architecture'] == 'cnn':
            model_name = model_name_base_cnn + f'_specId={spec_id}_mask={spec["mask"]}_pooltype={spec["pool_type"]}'
        else:
            model_name = model_name_base_cnn_another + f'_specId={spec_id}_mask={spec["mask"]}_pooltype={spec["pool_type"]}'

        skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
        oof_preds_optimized = np.zeros(seq_train.shape[0])
        oof_preds_majority = np.zeros(seq_train.shape[0])
        results = []
        for fold, (index_train, index_valid) in enumerate(skf.split(label_train, label_train)):
            logger.info(f'Fold {fold + 1} / {KFOLD} - create dataloader and build model')
            x_train, x_valid = seq_train[index_train].astype(int), seq_train[index_valid].astype(int)
            y_train, y_valid = label_train[index_train].astype(np.float32), label_train[index_valid].astype(np.float32)

            if spec['architecture'] == 'rnn':
                model = AttentionMaskRNNAnother(embedding_matrix, hidden_size=64, out_hidden_dim=64, embed_drop=0.2,
                                                out_drop=0.3, attention_type='dot', mask=spec['mask'])
            elif spec['architecture'] == 'cnn':
                model = BranchedMaskCNNRNN(embedding_matrix, hidden_size=64, out_hidden_dim=64, embed_drop=0.2,
                                           out_drop=0.3, attention_type='dot', mask=spec['mask'],
                                           pool_type=spec['pool_type'], kernel_sizes=(2, 7))
            else:
                model = BranchedMaskCNNRNNAnother(embedding_matrix, hidden_size=64, out_hidden_dim=64, embed_drop=0.2,
                                                  out_drop=0.3, attention_type='dot', mask=spec['mask'],
                                                  pool_type=spec['pool_type'], kernel_sizes=(2, 7))

            config = {
                'epochs': epochs,
                'batch_size': batch_size,
                'output_device': output_device,
                'criterion_type': 'bce',
                'criteria_weights': [0.5, 0.5],
                'criterion_gamma': 2.0,
                'criterion_alpha': 0.75,
                'optimizer': 'adam',
                'optimizer_lr': 0.003,
                'num_snapshots': NUM_SNAPSHOTS,
                'scheduler_type': 'cyclic',
                'base_lr': 0.00001,
                'max_lr': 0.003,
                'step_size': 1200,
                'scheduler_mode': 'triangular',
                'scheduler_gamma': 0.9,
                'scheduler_trigger_steps': 4000,
                'sampler_type': 'normal',
                'seed': SEED
            }

            trainer = Trainer(model, logger, config)
            eval_results = trainer.train_and_eval_fold(x_train, y_train, x_valid, y_valid, fold)

            oof_preds_majority[index_valid] = np.array([res['preds_binary'] for res in eval_results]).mean(0) > 0.5
            oof_majority_f1 = f1_score(label_train.reshape(-1,)[index_valid], oof_preds_majority[index_valid])

            oof_preds_proba = np.array([res['preds_proba'] for res in eval_results]).mean(0)
            oof_threshold_mean: float = np.mean([res['best_threshold'] for res in eval_results])
            oof_preds_optimized[index_valid] = oof_preds_proba > oof_threshold_mean
            oof_optimized_f1 = f1_score(label_train.reshape(-1,)[index_valid], oof_preds_optimized[index_valid])

            message = f'Fold {fold + 1} / {KFOLD} has been done.\n'
            message += f'Score: majority voting - {oof_majority_f1:.6f}, optimized threshold - {oof_optimized_f1:.6f}'
            logger.post(message)

            post_to_snapshot_spreadsheet(logger, SPREADSHEET_SNAPSHOT_URL, eval_type='SNAPSHOT', tag='SCORE',
                                         script_name=SCRIPT_NAME, model_name=model_name, fold=fold,
                                         snapshot_info=[res['f1'] for res in eval_results])

            post_to_snapshot_spreadsheet(logger, SPREADSHEET_SNAPSHOT_URL, eval_type='SNAPSHOT', tag='THRESHOLD',
                                         script_name=SCRIPT_NAME, model_name=model_name, fold=fold,
                                         snapshot_info=[res['best_threshold'] for res in eval_results])

            post_to_main_spreadsheet(logger, SPREADSHEET_MAIN_URL, eval_type='SNAPSHOT', script_name=SCRIPT_NAME,
                                     model_name=model_name, fold=fold, f1_majority=oof_majority_f1,
                                     f1_optimized=oof_optimized_f1, threshold=oof_threshold_mean)

            results.append({
                'f1_majority': oof_majority_f1,
                'f1_optimized': oof_optimized_f1,
                'threshold': oof_threshold_mean
            })

        f1_majority_mean = np.mean([res['f1_majority'] for res in results])
        f1_majority_std = np.std([res['f1_majority'] for res in results])
        f1_optimized_mean = np.mean([res['f1_optimized'] for res in results])
        f1_optimized_std = np.std([res['f1_optimized'] for res in results])
        threshold_mean = np.mean([res['threshold'] for res in results])
        total_metrics = [
            f1_majority_mean,
            f1_majority_std,
            f1_optimized_mean,
            f1_optimized_std,
            threshold_mean
        ]

        post_to_main_spreadsheet(logger, SPREADSHEET_MAIN_URL, eval_type='SNAPSHOT', script_name=SCRIPT_NAME,
                                 model_name=model_name, fold=-1, f1_majority=-1,
                                 f1_optimized=-1, threshold=-1, others=total_metrics)

        message = 'KFold training and evaluation has been done.\n'
        message += f'F1 majority voting - Avg: {f1_majority_mean}, Std: {f1_majority_std}\n'
        message += f'F1 optimized - Avg: {f1_optimized_mean}, Std: {f1_optimized_std}\n'
        message += f'Threshold - Avg: {threshold_mean}'
        logger.post(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--device-ids', metavar='N', type=int, nargs='+', default=[0])
    parser.add_argument('--max-workers', default=2, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args().__dict__

    log_dir = LOG_DIR.joinpath(f'{SCRIPT_NAME}')
    logger = Logger(SCRIPT_NAME, log_dir=log_dir, webhook_url=SLACK_URL, overwrite=True)
    logger.info(f'Script starts with command line arguments: {args}')
    try:
        main(logger, args)
        logger.post('===== Script completed successfully! =====')
    except Exception as e:
        logger.exception(e)
