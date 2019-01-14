from typing import List, Tuple, Dict, Union, Callable
import os
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader

from .logger import Logger
from .loss import L2Regulaization


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
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


def train_model(model: nn.Module, criteria: List[List], metric: Callable, optimizer: Optimizer,
                scheduler: Union[None, _LRScheduler, ReduceLROnPlateau], dataloaders: Dict[str, DataLoader],
                logger: Logger, config: dict) -> Tuple[nn.Module, float]:
    epochs = config['epochs']
    loss_names = config['loss_names']
    metric_type = config['metric_type']
    fold = config['fold']
    model_save_path = config['model_save_path']
    mode = config['mode']  # min or max
    early_stopping = config['early_stopping']
    model_name = config['model_name']

    train_dataloader = dataloaders['train']
    valid_dataloader = dataloaders['valid']

    logger.add_weight_histogram(model, 0, model_name)

    if mode == 'max':
        metric_best = - 999999.9
    else:
        metric_best = 999999.9
    loss_history = []
    metric_history = []
    epoch_best = 0
    early_stopping_counts = 0
    for epoch in range(1, epochs + 1):
        logger.info(f'Epoch {epoch:d} / {epochs:d}')
        logger.info(f'Training phase starts.\n')
        model, train_losses, train_metric = train_on_epoch(model, criteria, metric, metric_type,
                                                           optimizer, train_dataloader, config)
        logger.info(f'Training phase has been done. Validation phase starts.\n')
        model, valid_losses, valid_metric = validate_on_epoch(model, criteria, metric, metric_type,
                                                              valid_dataloader, config)

        message = f'Validation phase has been done. Epoch {epoch:d} / {epochs:d} results:\n'
        message += 'Train - ' + ', '.join([f'{name}: {loss:.5f}' for name, loss in zip(loss_names, train_losses)])
        message += f', metric: {train_metric:.5f}\n'
        message += 'Validation - ' + ', '.join([f'{name}: {loss:.5f}' for name, loss in zip(loss_names, valid_losses)])
        message += f', metric: {valid_metric:.5f}\n'
        logger.info(message)

        for loss_name, train_loss, valid_loss in zip(loss_names, train_losses, valid_losses):
            logger.add_scalars(loss_name, {f'train_{fold}': train_loss, f'valid_{fold}': valid_loss}, epoch)

        logger.add_scalars('metric', {'train': train_metric, 'valid': valid_metric}, epoch)
        logger.add_weight_histogram(model, epoch, model_name)

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(metrics=valid_metric)
            else:
                scheduler.step()

        loss_history.append([np.sum(train_losses), np.sum(valid_losses)])
        metric_history.append([train_metric, valid_metric])

        condition = valid_metric < metric_best if mode == 'min' else valid_metric > metric_best
        if condition:
            message = f'Validation metric improved from {metric_best:.5f} to -> {valid_metric:.5f}.\n'
            logger.info(message)
            if model_save_path:
                message = f'The model is being saved to -> {model_save_path}.\n'
                logger.info(message)
                if Path(model_save_path).exists():
                    Path(model_save_path).unlink()
                torch.save(model.state_dict(), model_save_path)
            metric_best = valid_metric
            epoch_best = epoch
            early_stopping_counts = 0
        else:
            early_stopping_counts += 1

        if early_stopping_counts >= early_stopping:
            message = f"""
            Early stopping at epoch {epoch} since the validation metric 
            didn't improve for consequent {early_stopping} epochs.
            """
            logger.info(message)
            break

    message = f'Training is done. Total Epochs: {epoch:d}\n'
    message += f'Best Validation Metric: {metric_best:.5f} at Epoch {epoch_best:d}.'
    logger.post(message)

    return model, metric_best


def train_on_epoch(model: nn.Module, criteria: List[List], metric: Callable, metric_type: str,
                   optimizer: Optimizer, dataloader: DataLoader, config: dict) -> Tuple[nn.Module, np.ndarray, float]:
    output_device = config['output_device']
    reg_lambda = config['reg_lambda']

    train_losses = np.zeros(len(criteria[0]))
    train_metric = 0
    outputs_total = []
    targets_total = []
    n_iter = len(dataloader)
    model.train()
    with tqdm(total=n_iter) as progress_bar:
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
                train_metric += metric(outputs, ) / n_iter
            else:
                outputs_total.append(to_cpu(outputs))
                targets_total.append(to_cpu(targets))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            progress_bar.update(1)

    if metric_type != 'cumulative':
        outputs_total = np.concatenate(outputs_total, axis=0)
        targets_total = np.concatenate(targets_total, axis=0)
        train_metric = metric(outputs_total, targets_total)

    return model, train_losses, train_metric


def validate_on_epoch(model: nn.Module, criteria: List[List], metric: Callable, metric_type: str,
                      dataloader: DataLoader, config: dict) -> Tuple[nn.Module, np.ndarray, float]:
    output_device = config['output_device']

    valid_losses = np.zeros(len(criteria[0]))
    valid_metric = 0
    outputs_total = []
    targets_total = []
    n_iter = len(dataloader)
    model.eval()
    with torch.no_grad(), tqdm(total=n_iter) as progress_bar:
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = to_gpu(inputs, output_device)
            targets = to_gpu(targets, output_device)
            outputs = model(inputs)

            for j, (criterion, coeff) in enumerate(zip(criteria[0], criteria[1])):
                loss = coeff * criterion(outputs, targets)
                valid_losses[j] += loss.item() / n_iter

            if metric_type == 'cumulative':
                valid_metric += metric(outputs, targets) / n_iter
            else:
                outputs_total.append(to_cpu(outputs))
                targets_total.append(to_cpu(targets))
            progress_bar.update(1)

    if metric_type != 'cumulative':
        outputs_total = np.concatenate(outputs_total, axis=0)
        targets_total = np.concatenate(targets_total, axis=0)
        valid_metric = metric(outputs_total, targets_total)

    return model, valid_losses, valid_metric


def predict(model: nn.Module, dataloader: DataLoader, config: dict) -> Union[np.ndarray, Dict, List]:
    output_device = config['output_device']
    model_save_path = config['model_save_path']
    model.load_state_dict(torch.load(model_save_path))
    results = []
    model.eval()
    with torch.no_grad(), tqdm(total=len(dataloader)) as progress_bar:
        for i, (inputs, _) in enumerate(dataloader):
            inputs = to_gpu(inputs, output_device)
            outputs = model.forward(inputs)
            outputs = to_cpu(outputs)
            results.append(outputs)
            progress_bar.update(1)

    if isinstance(results[0], np.ndarray):
        return np.concatenate(results, axis=0)
    elif isinstance(results[0], dict):
        return dict([(key, [d[key] for d in results]) for key in results[0].keys()])
    else:
        return results
