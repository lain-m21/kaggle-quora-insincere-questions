import sys; sys.path.append('..')
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences

import torch

from utils.logger import Logger, post_to_main_spreadsheet, post_to_snapshot_spreadsheet
from utils.prepare_data import load_data, tokenize_text, load_multiple_embeddings, preprocess_text, extract_nlp_features
from utils.trainer import Trainer

from model.nlprnn import NLPFeaturesRNN


INPUT_DIR = Path.cwd().joinpath('../../input')
DATA_DIR = Path.cwd().joinpath('../../data')
LOG_DIR = Path.cwd().joinpath('../../logs')
SLACK_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['slack']
SPREADSHEET_MAIN_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['spreadsheet_main']
SPREADSHEET_SNAPSHOT_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['spreadsheet_snapshots']

SCRIPT_NAME = Path(__file__).stem

SEED = 42
PADDING_LENGTH = 70
EPOCHS = 5
TRIGGER = 1
NUM_SNAPSHOTS = 5
KFOLD = 5


def main(logger, args):
    df_train, _ = load_data(INPUT_DIR, logger)
    if args['debug']:
        df_train = df_train.iloc[:200000]
        logger.info('Extract nlp features')
        df_train = extract_nlp_features(df_train)
    else:
        logger.info('Extract nlp features')
        df_train = extract_nlp_features(df_train)
        logger.info('Preprocess text')
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

    continuous_columns = ['total_length', 'n_capitals', 'n_words', 'n_puncts', 'n_?', 'n_!', 'n_you']
    for col in continuous_columns:
        scaler = StandardScaler()
        df_train[col] = scaler.fit_transform(df_train[col].values.astype(np.float32).reshape(-1, 1)).reshape(-1, )

    x_continuous = [df_train[col].values.reshape(-1, 1) for col in continuous_columns]

    # ===== training and evaluation loop ===== #
    device_ids = args['device_ids']
    output_device = device_ids[0]
    torch.cuda.set_device(device_ids[0])
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    batch_size = args['batch_size'] * len(device_ids)
    epochs = EPOCHS
    trigger = TRIGGER

    logger.info('Start training and evaluation loop')

    model_specs = [{'nlp_dim': 64, 'nlp_dropout': 0.2, 'activate': 'relu', 'factorize': False},
                   {'nlp_dim': 64, 'nlp_dropout': 0.2, 'activate': 'prelu', 'factorize': False},
                   {'nlp_dim': 64, 'nlp_dropout': 0.2, 'activate': 'relu', 'factorize': True},
                   {'nlp_dim': 128, 'nlp_dropout': 0.2, 'activate': 'relu', 'factorize': False},
                   {'nlp_dim': 128, 'nlp_dropout': 0.3, 'activate': 'relu', 'factorize': False},
                   {'nlp_dim': 128, 'nlp_dropout': 0.5, 'activate': 'relu', 'factorize': False},
                   {'nlp_dim': 64, 'nlp_dropout': 0.2, 'activate': 'prelu', 'factorize': True},
                   {'nlp_dim': 128, 'nlp_dropout': 0.2, 'activate': 'prelu', 'factorize': False},
                   {'nlp_dim': 128, 'nlp_dropout': 0.5, 'activate': 'prelu', 'factorize': False},
                   {'nlp_dim': 128, 'nlp_dropout': 0.5, 'activate': 'prelu', 'factorize': True}]

    model_name_base = 'NLPFeaturesRNN'

    for spec_id, spec in enumerate(model_specs):
        model_name = model_name_base + f'_specId={spec_id}_nlpdim={spec["nlp_dim"]}_nlpdrop={spec["nlp_dropout"]}'
        model_name += f'_activate={spec["activate"]}_factorize={spec["factorize"]}'

        skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
        oof_preds_optimized = np.zeros(len(seq_train))
        oof_preds_majority = np.zeros(len(seq_train))
        results = []
        for fold, (index_train, index_valid) in enumerate(skf.split(label_train, label_train)):
            logger.info(f'Fold {fold + 1} / {KFOLD} - create dataloader and build model')
            x_train = {
                'text': seq_train[index_train].astype(int),
                'continuous': [x[index_train] for x in x_continuous]
            }
            x_valid = {
                'text': seq_train[index_valid].astype(int),
                'continuous': [x[index_valid] for x in x_continuous]
            }
            y_train, y_valid = label_train[index_train].astype(np.float32), label_train[index_valid].astype(np.float32)

            model = NLPFeaturesRNN({'continuous': len(x_continuous)}, embedding_matrix, PADDING_LENGTH,
                                   hidden_size=64, out_hidden_dim=64, out_drop=0.3, embed_drop=0.2,
                                   dense_activate=spec['activate'], nlp_hidden_dim=spec['nlp_dim'], mask=False,
                                   nlp_dropout=spec['nlp_dropout'], factorize=spec['factorize'], embed_drop_direction=0)

            steps_per_epoch = seq_train[index_train].shape[0] // batch_size
            scheduler_trigger_steps = steps_per_epoch * trigger
            step_size = steps_per_epoch * (epochs - trigger) // NUM_SNAPSHOTS

            config = {
                'epochs': epochs,
                'batch_size': batch_size,
                'output_device': output_device,
                'criterion_type': 'bce',
                'criteria_weights': [1.0, 1.0],
                'criterion_gamma': 2.0,
                'criterion_alpha': 0.75,
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
