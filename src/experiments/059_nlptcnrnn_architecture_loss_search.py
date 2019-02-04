import sys; sys.path.append('..')
import json
import argparse
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras.preprocessing.sequence import pad_sequences

import torch

from utils.logger import Logger
from utils.logger import post_to_total_metrics_table, post_to_fold_metrics_table, post_to_snapshot_metrics_table
from utils.prepare_data import load_data, tokenize_texts, load_multiple_embeddings, preprocess_text, extract_nlp_features
from utils.trainer import Trainer
from utils.metrics import calculate_fold_metrics, calculate_total_metrics

from model.tcnrnn import NLPFeaturesTCNRNN


INPUT_DIR = Path.cwd().joinpath('../../input')
DATA_DIR = Path.cwd().joinpath('../../data')
LOG_DIR = Path.cwd().joinpath('../../logs')
SLACK_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['slack']
BQ_PROJECT_ID = json.load(Path.cwd().joinpath('../config.json').open('r'))['bq_project_id']
BQ_DATASET = json.load(Path.cwd().joinpath('../config.json').open('r'))['bq_dataset']
SPREADSHEET_MAIN_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['spreadsheet_main']
SPREADSHEET_SNAPSHOT_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['spreadsheet_snapshots']

SCRIPT_NAME = Path(__file__).stem

SEED = 42
PADDING_LENGTH = 70
EPOCHS = 5
TRIGGER = 1
NUM_SNAPSHOTS = 5
KFOLD = 5


def parallel_apply(func_and_args):
    func, func_args = func_and_args
    return func(*func_args)


def main(logger, args):
    df_train, _ = load_data(INPUT_DIR, logger)
    if args['debug']:
        df_train = df_train.iloc[:30000]
        texts_train = df_train['question_text']
    else:
        logger.info('Preprocess text')
        texts_train = preprocess_text(df_train, return_df=False)
    seq_train, tokenizer = tokenize_texts(texts_train, logger)
    logger.info('Pad train text data')
    seq_train = pad_sequences(seq_train, maxlen=PADDING_LENGTH)

    label_train = df_train['target'].values.reshape(-1, 1)

    embed_types = [0, 1, 2]

    logger.info('Start multiprocess nlp feature extraction and embedding matrices loading')
    with mp.Pool(processes=2) as p:
        results = p.map(parallel_apply, [
            (extract_nlp_features, (df_train,)),
            (load_multiple_embeddings, (tokenizer.word_index, embed_types, args['debug']))
        ])

    df_train_extracted = results[0]
    embedding_matrices = results[1]
    embedding_matrix = np.concatenate(
        [np.array([embedding_matrices[i] for i in [0, 1, 2]]).mean(0)] + [
            embedding_matrices[j] for j in [1]
        ], axis=1
    )

    nlp_columns = ['total_length', 'n_capitals', 'n_words', 'n_puncts', 'n_?', 'n_!', 'n_you']
    for col in nlp_columns:
        scaler = StandardScaler()
        df_train_extracted[col] = scaler.fit_transform(
            df_train_extracted[col].values.astype(np.float32).reshape(-1, 1)).reshape(-1, )

    x_nlp = [df_train_extracted[col].values.reshape(-1, 1) for col in nlp_columns]
    nlp_size = len(x_nlp)

    # ===== training and evaluation loop ===== #
    device_ids = args['device_ids']
    output_device = device_ids[0]
    torch.cuda.set_device(device_ids[0])
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    batch_size = args['batch_size'] * len(device_ids)
    trigger = TRIGGER

    if args['debug']:
        epochs = 3
        n_splits = 2
    else:
        epochs = EPOCHS
        n_splits = KFOLD

    logger.info('Start training and evaluation loop')

    model_specs = [
        {'nlp_layer_types': ({'activation': 'relu', 'dim': 16, 'dropout': 0.2},
                             {'activation': 'relu', 'dim': 16, 'dropout': 0.2}),
         'tcn_layer_types': ({'num_channels': [16, 16, 16], 'num_layers': 1, 'kernel_size': 3, 'dropout': 0.2},),
         'rnn_layer_types': ({'type': 'lstm', 'dim': 64, 'num_layers': 1, 'dropout': 0.0},
                             {'type': 'gru', 'dim': 64, 'num_layers': 1, 'dropout': 0.0}),
         'upper_layer_types': ({'dim': 64, 'dropout': 0.5},
                               {'dim': 64, 'dropout': 0.5},
                               {'dim': 64, 'dropout': 0.3}),
         'gamma': 2.0, 'alpha': 0.75, 'combined': False, 'focal': False, 'weight': 1.0, 'mask': True},
        {'nlp_layer_types': ({'activation': 'relu', 'dim': 16, 'dropout': 0.2},
                             {'activation': 'relu', 'dim': 16, 'dropout': 0.2}),
         'tcn_layer_types': ({'num_channels': [32, 32, 32], 'num_layers': 1, 'kernel_size': 3, 'dropout': 0.2},),
         'rnn_layer_types': ({'type': 'lstm', 'dim': 64, 'num_layers': 1, 'dropout': 0.0},
                             {'type': 'gru', 'dim': 64, 'num_layers': 1, 'dropout': 0.0}),
         'upper_layer_types': ({'dim': 64, 'dropout': 0.5},
                               {'dim': 64, 'dropout': 0.5},
                               {'dim': 64, 'dropout': 0.3}),
         'gamma': 2.0, 'alpha': 0.75, 'combined': False, 'focal': False, 'weight': 1.0, 'mask': True},
        {'nlp_layer_types': ({'activation': 'relu', 'dim': 16, 'dropout': 0.2},
                             {'activation': 'relu', 'dim': 16, 'dropout': 0.2}),
         'tcn_layer_types': ({'num_channels': [64, 64, 64], 'num_layers': 1, 'kernel_size': 3, 'dropout': 0.2},),
         'rnn_layer_types': ({'type': 'lstm', 'dim': 64, 'num_layers': 1, 'dropout': 0.0},
                             {'type': 'gru', 'dim': 64, 'num_layers': 1, 'dropout': 0.0}),
         'upper_layer_types': ({'dim': 64, 'dropout': 0.5},
                               {'dim': 64, 'dropout': 0.5},
                               {'dim': 64, 'dropout': 0.3}),
         'gamma': 2.0, 'alpha': 0.75, 'combined': False, 'focal': False, 'weight': 1.0, 'mask': True},
        {'nlp_layer_types': ({'activation': 'relu', 'dim': 16, 'dropout': 0.2},
                             {'activation': 'relu', 'dim': 16, 'dropout': 0.2}),
         'tcn_layer_types': ({'num_channels': [32, 32, 32], 'num_layers': 1, 'kernel_size': 3, 'dropout': 0.2},),
         'rnn_layer_types': ({'type': 'lstm', 'dim': 64, 'num_layers': 1, 'dropout': 0.0},
                             {'type': 'gru', 'dim': 64, 'num_layers': 1, 'dropout': 0.0}),
         'upper_layer_types': ({'dim': 64, 'dropout': 0.5},
                               {'dim': 64, 'dropout': 0.5},
                               {'dim': 64, 'dropout': 0.3}),
         'gamma': 2.0, 'alpha': 0.75, 'combined': True, 'focal': True, 'weight': 5.0, 'mask': True},
    ]
    model_name_base = 'NLPFeaturesTCNRNN'

    for spec_id, spec in enumerate(model_specs):
        model_name = model_name_base + f'_specId={spec_id}'

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        oof_mv_preds = np.zeros(len(seq_train))
        oof_preds_proba = np.zeros(len(seq_train))
        oof_opt_preds = np.zeros(len(seq_train))
        oof_reopt_preds = np.zeros(len(seq_train))
        results_list = []
        for fold, (index_train, index_valid) in enumerate(skf.split(label_train, label_train)):
            logger.info(f'Fold {fold + 1} / {KFOLD} - create dataloader and build model')
            x_train = {
                'text': seq_train[index_train].astype(int),
                'nlp': [x[index_train] for x in x_nlp]
            }
            x_valid = {
                'text': seq_train[index_valid].astype(int),
                'nlp': [x[index_valid] for x in x_nlp]
            }
            y_train, y_valid = label_train[index_train].astype(np.float32), label_train[index_valid].astype(np.float32)

            model = NLPFeaturesTCNRNN(embedding_matrix, PADDING_LENGTH, nlp_size,
                                      embed_drop=0.2, mask=spec['mask'],
                                      nlp_layer_types=spec['nlp_layer_types'],
                                      tcn_layer_types=spec['tcn_layer_types'],
                                      rnn_layer_types=({'type': 'lstm', 'dim': 64, 'num_layers': 1, 'dropout': 0.0},
                                                       {'type': 'gru', 'dim': 64, 'num_layers': 1, 'dropout': 0.0}),
                                      upper_layer_types=spec['upper_layer_types'])

            steps_per_epoch = seq_train[index_train].shape[0] // batch_size
            scheduler_trigger_steps = steps_per_epoch * trigger
            step_size = steps_per_epoch * (epochs - trigger) // NUM_SNAPSHOTS

            if spec['focal']:
                if spec['combined']:
                    criterion_type = 'bce_focal'
                else:
                    criterion_type = 'focal'
            else:
                criterion_type = 'bce'

            config = {
                'epochs': epochs,
                'batch_size': batch_size,
                'output_device': output_device,
                'criterion_type': criterion_type,
                'criteria_weights': [1.0, spec['weight']],
                'criterion_gamma': spec['gamma'],
                'criterion_alpha': spec['alpha'],
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

            fold_results = calculate_fold_metrics(eval_results, label_train[index_valid].reshape(-1,))
            results_list.append(fold_results)

            message = f'Fold {fold + 1} / {KFOLD} has been done.\n'

            message += f'Majority Voting - F1: {fold_results["oof_mv_f1"]}, '
            message += f'Precision: {fold_results["oof_mv_precision"]}, Recall: {fold_results["oof_mv_recall"]}\n'

            message += f'Optimized - F1: {fold_results["oof_opt_f1"]}, '
            message += f'Precision: {fold_results["oof_opt_precision"]}, Recall: {fold_results["oof_opt_recall"]}\n'

            message += f'Re-optimized - F1: {fold_results["oof_reopt_f1"]}, '
            message += f'Precision: {fold_results["oof_reopt_precision"]}, Recall: {fold_results["oof_reopt_recall"]}\n'

            message += f'Focal Loss: {fold_results["oof_focal_loss"]}, '
            message += f'Optimized Threshold: {fold_results["oof_opt_threshold"]}, '
            message += f'Re-optimized Threshold: {fold_results["oof_reopt_threshold"]}, '
            logger.post(message)

            eval_results_addition = {
                'date': datetime.now(),
                'script_name': SCRIPT_NAME,
                'spec_id': spec_id,
                'model_name': model_name,
                'fold_id': fold
            }
            for res in eval_results:
                res.update(eval_results_addition)
                # post_to_snapshot_metrics_table(data=res, project_id=BQ_PROJECT_ID, dataset_name=BQ_DATASET)

            fold_results_addition = {
                'date': datetime.now(),
                'script_name': SCRIPT_NAME,
                'spec_id': spec_id,
                'model_name': model_name,
                'fold_id': fold
            }
            fold_results.update(fold_results_addition)
            post_to_fold_metrics_table(fold_results, project_id=BQ_PROJECT_ID, dataset_name=BQ_DATASET)

            oof_mv_preds[index_valid] = fold_results['oof_mv_preds']
            oof_opt_preds[index_valid] = fold_results['oof_opt_preds']
            oof_reopt_preds[index_valid] = fold_results['oof_reopt_preds']
            oof_preds_proba[index_valid] = fold_results['oof_preds_proba']

        results = calculate_total_metrics(results_list)

        results_addition = {
            'date': datetime.now(),
            'script_name': SCRIPT_NAME,
            'spec_id': spec_id,
            'model_name': model_name
        }
        results.update(results_addition)

        if args['save_preds']:
            save_path = DATA_DIR.joinpath(f'predictions/{SCRIPT_NAME + "_" + model_name + ".pkl"}')
            predictions = {
                'proba': oof_preds_proba,
                'mv': oof_mv_preds,
                'opt': oof_opt_preds,
                'reopt': oof_reopt_preds
            }
            joblib.dump(predictions, str(save_path))

        post_to_total_metrics_table(results, project_id=BQ_PROJECT_ID, dataset_name=BQ_DATASET)

        logger.post(f'Spec ID: {spec_id}\nModel Spec: {spec}')

        message = 'KFold training and evaluation has been done.\n'
        message += f'Majority Voting - F1: avg = {results["mv_f1_avg"]}, std = {results["mv_f1_std"]}, '
        message += f'Precision: {results["mv_precision_avg"]}, Recall: {results["mv_recall_avg"]}\n'

        message += f'Optimized - F1: avg = {results["opt_f1_avg"]}, std = {results["opt_f1_std"]}, '
        message += f'Precision: {results["opt_precision_avg"]}, Recall: {results["opt_recall_avg"]}\n'

        message += f'Re-optimized - F1: avg = {results["reopt_f1_avg"]}, std = {results["reopt_f1_std"]}, '
        message += f'Precision: {results["reopt_precision_avg"]}, Recall: {results["reopt_recall_avg"]}\n'

        mv_thresholds = ", ".join([str(th) for th in results["mv_thresholds_avg"]])

        message += f'Focal Loss: {results["focal_loss_avg"]}, '
        message += f'Optimized Threshold: {results["opt_threshold_avg"]}, '
        message += f'Re-optimized Threshold: {results["reopt_threshold_avg"]}\n'
        message += f'Majority Voting Thresholds: {mv_thresholds}'
        logger.post(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--device-ids', metavar='N', type=int, nargs='+', default=[0])
    parser.add_argument('--max-workers', default=2, type=int)
    parser.add_argument('--save-preds', action='store_true')
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
