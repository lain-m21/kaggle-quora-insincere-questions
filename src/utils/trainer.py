import os
import time
import datetime
from tqdm import tqdm
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .loaders import SimpleDataset, DictDataset, BinaryOverSampler, BinaryBalancedSampler, worker_init_fn
from .metrics import f1_from_logits_with_threshold
from .loss import SmoothF1Loss, FocalLoss


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

        self.criterion = None
        self.criteria = None
        if config['criterion_type'] == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif config['criterion_type'] == 'f1':
            self.criterion = SmoothF1Loss(logit=True)
        elif config['criterion_type'] == 'focal':
            self.criterion = FocalLoss(logit=True, gamma=config['criterion_gamma'], alpha=config['criterion_alpha'])
        elif config['criterion_type'] == 'bce_f1':
            self.criteria = [[nn.BCEWithLogitsLoss(reduction='mean'), SmoothF1Loss(logit=True)],
                             config['criteria_weights']]
        elif config['criterion_type'] == 'bce_focal':
            self.criteria = [[nn.BCEWithLogitsLoss(reduction='mean'),
                              FocalLoss(logit=True, gamma=config['criterion_gamma'], alpha=config['criterion_alpha'])],
                             config['criteria_weights']]
        else:
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

    def train_and_eval_fold(self, x_train, y_train, x_valid, y_valid, fold_idx):
        train_loader = self.get_loader(x_train, y_train, mode='train')
        valid_loader = self.get_loader(x_valid, y_valid, mode='valid')

        n_iter = len(train_loader)
        checkpoint_count = 0
        step_count = 0
        start_time = time.time()

        self.logger.info(f'Start training and evaluation loop')

        eval_results = []
        for epoch in range(self.epochs):
            self.model.train()
            loss_epoch = 0
            self.logger.info(f'Epoch: {epoch + 1} / {self.epochs}')
            with tqdm(total=n_iter) as progress_bar:
                for i, (inputs, targets) in tqdm(enumerate(train_loader)):
                    inputs, targets = self._tensors_to_gpu(inputs), self._tensors_to_gpu(targets)

                    outputs = self.model(inputs)
                    if self.criterion is not None:
                        loss = self.criterion(outputs, targets)
                    else:
                        loss = 0
                        criteria = self.criteria[0]
                        weights = self.criteria[1]
                        for criterion, weight in zip(criteria, weights):
                            loss += weight * criterion(outputs, targets)

                    loss_epoch += loss.item() / n_iter

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step(closure=None)
                    if self.scheduler_type == 'cyclic' and step_count > self.scheduler_trigger_steps:
                        checkpoint_flag = self.scheduler.batch_step()
                        if checkpoint_flag:
                            self.logger.info('Cyclic scheduler hit the bottom. Start evaluation.')
                            eval_result = self.evaluate(valid_loader)
                            eval_result['epoch'] = epoch
                            eval_result['steps'] = step_count
                            eval_result['fold'] = fold_idx
                            eval_results.append(eval_result)

                            elapsed = str(datetime.timedelta(seconds=time.time() - start_time))
                            message = f'Fold: {fold_idx}, Epoch: {epoch + 1} / {self.epochs}, Steps: {step_count} / {n_iter}, '
                            message += f'Checkpoint: {checkpoint_count}, Train Loss: {loss_epoch}, '
                            message += f'Eval Loss: {eval_result["loss"]}, F1: {eval_result["f1"]}, '
                            message += f'Best threshold: {eval_result["best_threshold"]}, Elapsed: {elapsed} sec'
                            self.logger.info(message)

                            checkpoint_count += 1
                            if checkpoint_count >= self.num_snapshots:
                                break
                            self.model.train()

                    step_count += 1
                    progress_bar.update(1)

            if self.scheduler_type == 'step':
                self.scheduler.step()
                eval_result = self.evaluate(valid_loader)
                eval_result['epoch'] = epoch
                eval_result['fold'] = fold_idx
                eval_results.append(eval_result)

                elapsed = str(datetime.timedelta(seconds=time.time() - start_time))
                message = f'Fold: {fold_idx}, Epoch: {epoch + 1} / {self.epochs}, Train Loss: {loss_epoch}, '
                message += f'Eval Loss: {eval_result["loss"]}, F1: {eval_result["f1"]}, '
                message += f'Best threshold: {eval_result["best_threshold"]}, Elapsed: {elapsed} sec'
                self.logger.info(message)

        return eval_results

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
            with tqdm(total=n_iter) as progress_bar:
                for i, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = self._tensors_to_gpu(inputs), self._tensors_to_gpu(targets)

                    outputs = self.model(inputs)
                    if self.criterion is not None:
                        loss = self.criterion(outputs, targets)
                    else:
                        loss = 0
                        criteria = self.criteria[0]
                        weights = self.criteria[1]
                        for criterion, weight in zip(criteria, weights):
                            loss += weight * criterion(outputs, targets)

                    loss_epoch += loss.item() / n_iter

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step(closure=None)
                    if self.scheduler_type == 'cyclic' and step_count > self.scheduler_trigger_steps:
                        checkpoint_flag = self.scheduler.batch_step()
                        if checkpoint_flag:
                            self.logger.info('Cyclic scheduler hit the bottom. Start evaluation.')
                            predict_results.append(self.predict(test_loader, thresholds[checkpoint_count]))
                            checkpoint_count += 1

                            if checkpoint_count >= self.num_snapshots:
                                break
                            self.model.train()

                    step_count += 1
                    progress_bar.update(1)

            if self.scheduler_type == 'step':
                self.scheduler.step()

        if self.scheduler_type != 'cyclic':
            predict_results.append(self.predict(test_loader, thresholds[0]))

        return predict_results

    def evaluate(self, valid_loader):
        total_outputs = []
        total_targets = []
        loss_eval = 0
        n_iter = len(valid_loader)

        self.model.eval()
        with torch.no_grad(), tqdm(total=n_iter) as progress_bar:
            for i, (inputs, targets) in tqdm(enumerate(valid_loader)):
                inputs, targets = self._tensors_to_gpu(inputs), self._tensors_to_gpu(targets)

                outputs = self.model(inputs)
                if self.criterion is not None:
                    loss = self.criterion(outputs, targets)
                else:
                    loss = 0
                    criteria = self.criteria[0]
                    weights = self.criteria[1]
                    for criterion, weight in zip(criteria, weights):
                        loss += weight * criterion(outputs, targets)

                loss_eval += loss.item() / n_iter

                total_outputs.append(self._tensors_to_numpy(outputs))
                total_targets.append(self._tensors_to_numpy(targets))

                progress_bar.update(1)

        total_outputs = np.concatenate(total_outputs, axis=0)
        total_targets = np.concatenate(total_targets, axis=0)
        threshold_search_result = f1_from_logits_with_threshold(total_outputs, total_targets)
        preds = sp.special.expit(total_outputs)
        eval_result = {
            'preds_proba': preds.reshape(-1,),
            'preds_binary': np.array(preds > threshold_search_result['threshold'], dtype=int).reshape(-1,),
            'best_threshold': threshold_search_result['threshold'],
            'f1': threshold_search_result['f1'],
            'loss': loss_eval
        }
        return eval_result

    def predict(self, test_loader, threshold):
        total_outputs = []
        n_iter = len(test_loader)

        self.model.eval()
        with torch.no_grad(), tqdm(total=n_iter) as progress_bar:
            for i, (inputs, _) in tqdm(enumerate(test_loader)):
                inputs = self._tensors_to_gpu(inputs)
                outputs = self.model(inputs)
                total_outputs.append(self._tensors_to_numpy(outputs))

                progress_bar.update(1)

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
                                    batch_size=self.batch_size * 2,
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
