from tqdm import tqdm
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score
import torch.nn as nn


def f1_from_logits(outputs: np.ndarray, targets: np.ndarray, threshold: float=0.5):
    outputs = outputs.reshape(-1,)
    targets = targets.reshape(-1,).astype(int)
    return f1_score(targets, sp.special.expit(outputs) > threshold)


def bce_from_logits(outputs, targets):
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    return bce_loss(outputs.detach(), targets.detach()).item()


def threshold_search(y_true, y_pred):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = f1_score(y_true, y_pred > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result
