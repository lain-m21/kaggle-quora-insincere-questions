import json
import time
import shutil
import requests
import traceback
from typing import Union
from pathlib import Path, PosixPath
from contextlib import contextmanager
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from logging import getLogger
from logging import StreamHandler
from logging import DEBUG
from logging import Formatter
from logging import FileHandler

import torch.nn as nn
from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, logger_name: str,
                 log_dir: Union[Path, PosixPath, str],
                 webhook_url: str,
                 overwrite: bool=True,
                 tensorboard: bool=True):

        self._logger = getLogger(logger_name)
        if log_dir.exists():
            if overwrite:
                shutil.rmtree(log_dir)
                log_dir.mkdir()
        else:
            log_dir.mkdir()

        log_fmt = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        streamhandler = StreamHandler()
        streamhandler.setLevel('INFO')
        streamhandler.setFormatter(log_fmt)
        self._logger.setLevel('INFO')
        self._logger.addHandler(streamhandler)

        filehandler = FileHandler(log_dir.joinpath('text_log.txt'), 'a')
        filehandler.setLevel(DEBUG)
        filehandler.setFormatter(log_fmt)
        self._logger.setLevel(DEBUG)
        self._logger.addHandler(filehandler)

        if tensorboard:
            self._writer = SummaryWriter(str(log_dir))
        else:
            self._writer = None
        self._log_step = 0
        self._url = webhook_url

    def info(self, message: str):
        self._logger.info(message)
        if self._writer:
            self._writer.add_text('INFO', message, self._log_step)
            self._log_step += 1

    def debug(self, message: str):
        self._logger.debug(message)
        if self._writer:
            self._writer.add_text('DEBUG', message, self._log_step)
            self._log_step += 1

    def post(self, message: str):
        self.info(message)
        if self._url:
            content = self._logger.name + ' - ' + message
            requests.post(self._url, json.dumps({'text': content}))
        else:
            raise ValueError('Slack URL not set!')

    def post_to_spreadsheet(self, data, url: str):
        requests.post(url, json.dumps(data))

    @contextmanager
    def timer(self, process_name: str):
        since = time.time()
        yield
        message = f'Process [{process_name}] finished in {time.time() - since:.2f} sec'
        self.info(message)

    def exception(self, e: Exception):
        message = ''.join(traceback.format_tb(e.__traceback__)) + '\n'
        message += str(e.__class__) + ' ' + str(e)
        self.post(message)
        raise e

    def add_scalar(self, tag: str, value: dict, step: int):
        assert self._writer is not None, 'Tensorboard SummaryWriter not set!'
        self._writer.add_scalar(tag, value, step)

    def add_scalars(self, tag: str, value_dict: dict, step: int):
        assert self._writer is not None, 'Tensorboard SummaryWriter not set!'
        self._writer.add_scalars(tag, value_dict, step)

    def add_histogram(self, tag: str, values: np.ndarray, step: int):
        assert self._writer is not None, 'Tensorboard SummaryWriter not set!'
        self._writer.add_histogram(tag, values, step)

    def add_weight_histogram(self, model: nn.Module, step: int, model_name: Union[None, str]=None):
        """
        Track records of model weights histograms. You can only use this method with a PyTorch model.
        """
        assert isinstance(model, nn.Module), 'model should be PyTorch nn.Module!'
        assert self._writer is not None, 'Tensorboard SummaryWriter not set!'
        if model_name is not None:
            prefix = model_name + '_'
        else:
            prefix = ''

        for name, param in model.named_parameters():
            self.add_histogram(prefix + name, param.clone().cpu().data.numpy(), step)


def post_to_main_spreadsheet(logger: Logger,
                             url: str,
                             eval_type: str,
                             script_name: str,
                             model_name: str,
                             fold: int,
                             f1_majority: float,
                             f1_optimized: float,
                             threshold: float,
                             others=None):
    if fold < 0:
        fold = 'total'
    else:
        fold = f'fold_{fold}'
    data = [eval_type, script_name, model_name, fold, f1_majority, f1_optimized, threshold]
    if others is not None:
        data.extend(others)
    logger.post_to_spreadsheet(data, url)


def post_to_snapshot_spreadsheet(logger: Logger,
                                 url: str,
                                 eval_type: str,
                                 tag: str,
                                 script_name: str,
                                 model_name: str,
                                 fold: int,
                                 snapshot_info):
    if fold < 0:
        fold = 'total'
    else:
        fold = f'fold_{fold}'
    data = [eval_type, tag, script_name, model_name, fold]
    data.extend(snapshot_info)
    logger.post_to_spreadsheet(data, url)


SCHEMA_MAP = {
    str: 'STRING',
    float: 'FLOAT',
    int: 'INTEGER'
}


def post_to_total_metrics_table(data, project_id, dataset_name):

    columns = [
        'date', 'script_name', 'spec_id', 'model_name', 'mv_f1_avg', 'mv_f1_std', 'mv_precision_avg', 'mv_recall_avg',
        'opt_f1_avg', 'opt_f1_std', 'opt_precision_avg', 'opt_recall_avg', 'reopt_f1_avg', 'reopt_f1_std',
        'reopt_precision_avg', 'reopt_recall_avg', 'focal_loss_avg', 'mv_threshold_avg_snapshot_0',
        'mv_threshold_avg_snapshot_1', 'mv_threshold_avg_snapshot_2', 'mv_threshold_avg_snapshot_3',
        'mv_threshold_avg_snapshot_4', 'opt_threshold_avg', 'reopt_threshold_avg'
    ]

    insert_data = dict([(col, [data[col]]) for col in columns])
    df = pd.DataFrame.from_dict(insert_data)

    schema = get_schema(df, columns)

    table_name = dataset_name + '.' + 'total_metrics'
    df.to_gbq(destination_table=table_name, project_id=project_id, if_exists='append', table_schema=schema)


def post_to_fold_metrics_table(data, project_id, dataset_name):

    columns = [
        'date', 'script_name', 'spec_id', 'model_name', 'fold_id', 'oof_mv_f1', 'oof_mv_precision', 'oof_mv_recall',
        'oof_opt_f1', 'oof_opt_precision', 'oof_opt_recall', 'oof_reopt_f1', 'oof_reopt_precision', 'oof_reopt_recall',
        'oof_focal_loss', 'oof_opt_threshold', 'oof_reopt_threshold'
    ]

    insert_data = dict([(col, [data[col]]) for col in columns])
    df = pd.DataFrame.from_dict(insert_data)

    schema = get_schema(df, columns)

    table_name = dataset_name + '.' + 'fold_metrics'
    df.to_gbq(destination_table=table_name, project_id=project_id, if_exists='append', table_schema=schema)


def post_to_snapshot_metrics_table(data, project_id, dataset_name):
    columns = [
        'date', 'script_name', 'spec_id', 'model_name', 'fold_id', 'snapshot_id',
        'snapshot_f1', 'snapshot_precision', 'snapshot_recall', 'snapshot_focal_loss', 'snapshot_threshold',
        'snapshot_epoch', 'snapshot_steps', 'snapshot_loss'
    ]

    insert_data = dict([(col, [data[col]]) for col in columns])
    df = pd.DataFrame.from_dict(insert_data)

    schema = get_schema(df, columns)

    table_name = dataset_name + '.' + 'snapshot_metrics'
    df.to_gbq(destination_table=table_name, project_id=project_id, if_exists='append', table_schema=schema)


def get_schema(df, columns):
    schema = []
    for col in columns:
        if col == 'date':
            schema.append({'name': 'date', 'type': 'DATETIME'})
            continue
        if type(df[col].item()) in SCHEMA_MAP:
            schema.append({'name': col, 'type': SCHEMA_MAP[type(df[col].item())]})
    return schema
