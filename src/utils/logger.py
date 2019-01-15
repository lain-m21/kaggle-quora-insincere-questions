import json
import time
import shutil
import requests
import traceback
from typing import Union
from pathlib import Path, PosixPath
from contextlib import contextmanager
import numpy as np

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
