from tqdm import tqdm
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn as nn


def f1_from_logits(outputs: np.ndarray, targets: np.ndarray, threshold: float=0.5):
    outputs = outputs.reshape(-1,)
    targets = targets.reshape(-1,).astype(int)
    return f1_score(targets, sp.special.expit(outputs) > threshold)


def f1_from_logits_optimized(outputs: np.ndarray, targets: np.ndarray, search_range=(0.3, 0.6), search_num=31):
    outputs = outputs.reshape(-1, )
    targets = targets.reshape(-1, ).astype(int)
    search_result = threshold_search(targets, sp.special.expit(outputs), search_range, search_num)
    return search_result['f1']


def f1_from_logits_with_threshold(outputs: np.ndarray, targets: np.ndarray, search_range=(0.25, 1.0), search_num=76):
    outputs = outputs.reshape(-1, )
    targets = targets.reshape(-1, ).astype(int)
    search_result = threshold_search(targets, sp.special.expit(outputs), search_range, search_num)
    return search_result


def bce_from_logits(outputs, targets):
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    return bce_loss(outputs.detach(), targets.detach()).item()


def focal_loss(outputs, targets, gamma=2.0, alpha=0.75, factor=100, epsilon=1e-10):
    outputs, targets = outputs.reshape(-1,), targets.reshape(-1,)
    outputs = np.clip(outputs, a_min=epsilon, a_max=1 - epsilon)

    p_t = targets * outputs + (1 - targets) * (1 - outputs)
    alpha = np.ones(len(outputs)) * alpha
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)

    cross_entropy = - (targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))
    weight = alpha_t * np.power((1 - p_t), gamma)
    loss = weight * cross_entropy
    loss = np.mean(loss) * factor
    return loss


def threshold_search(y_true, y_pred, search_range=(0.2, 0.8), search_num=61):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([th for th in np.linspace(*search_range, search_num)]):
        score = f1_score(y_true, y_pred > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


def calculate_snapshot_metrics(outputs, targets):
    snapshot_preds_proba = sp.special.expit(outputs)
    search_result = threshold_search(targets, snapshot_preds_proba)
    snapshot_threshold = search_result['threshold']
    snapshot_preds_binary = np.array(snapshot_preds_proba > snapshot_threshold, dtype=int)

    snapshot_f1 = f1_score(targets, snapshot_preds_binary)
    snapshot_precision = precision_score(targets, snapshot_preds_binary)
    snapshot_recall = recall_score(targets, snapshot_preds_binary)
    snapshot_focal_loss = focal_loss(snapshot_preds_proba, targets)

    results = {
        'snapshot_f1': snapshot_f1,
        'snapshot_precision': snapshot_precision,
        'snapshot_recall': snapshot_recall,
        'snapshot_focal_loss': snapshot_focal_loss,

        'snapshot_threshold': snapshot_threshold,

        'snapshot_preds_proba': snapshot_preds_proba,
        'snapshot_preds_binary': snapshot_preds_binary
    }
    return results


def calculate_fold_metrics(eval_results, targets):
    oof_mv_preds = np.array(np.array([res['snapshot_preds_binary'] for res in eval_results]).mean(0) > 0.5, dtype=int)
    oof_mv_thresholds = np.array([res['snapshot_threshold'] for res in eval_results])

    oof_mv_f1 = f1_score(targets, oof_mv_preds)
    oof_mv_precision = precision_score(targets, oof_mv_preds)
    oof_mv_recall = recall_score(targets, oof_mv_preds)

    oof_preds_proba = np.array([res['snapshot_preds_proba'] for res in eval_results]).mean(0)
    oof_opt_threshold = oof_mv_thresholds.mean()
    oof_opt_preds = np.array(oof_preds_proba > oof_opt_threshold, dtype=int)

    oof_opt_f1 = f1_score(targets, oof_opt_preds)
    oof_opt_precision = precision_score(targets, oof_opt_preds)
    oof_opt_recall = recall_score(targets, oof_opt_preds)
    oof_focal_loss = focal_loss(oof_preds_proba, targets)

    search_result = threshold_search(targets, oof_preds_proba)
    oof_reopt_threshold = search_result['threshold']
    oof_reopt_preds = np.array(oof_preds_proba > oof_reopt_threshold, dtype=int)

    oof_reopt_f1 = f1_score(targets, oof_reopt_preds)
    oof_reopt_precision = precision_score(targets, oof_reopt_preds)
    oof_reopt_recall = recall_score(targets, oof_reopt_preds)

    eval_results = {
        'oof_mv_f1': oof_mv_f1,
        'oof_mv_precision': oof_mv_precision,
        'oof_mv_recall': oof_mv_recall,

        'oof_opt_f1': oof_opt_f1,
        'oof_opt_precision': oof_opt_precision,
        'oof_opt_recall': oof_opt_recall,

        'oof_reopt_f1': oof_reopt_f1,
        'oof_reopt_precision': oof_reopt_precision,
        'oof_reopt_recall': oof_reopt_recall,

        'oof_focal_loss': oof_focal_loss,

        'oof_mv_thresholds': oof_mv_thresholds,
        'oof_opt_threshold': oof_opt_threshold,
        'oof_reopt_threshold': oof_reopt_threshold,

        'oof_mv_preds': oof_mv_preds,
        'oof_preds_proba': oof_preds_proba,
        'oof_opt_preds': oof_opt_preds,
        'oof_reopt_preds': oof_reopt_preds
    }
    return eval_results


def calculate_total_metrics(results_list):
    mv_f1_avg: float = np.mean([res['oof_mv_f1'] for res in results_list])
    mv_f1_std: float = np.std([res['oof_mv_f1'] for res in results_list])
    mv_precision_avg: float = np.mean([res['oof_mv_precision'] for res in results_list])
    mv_recall_avg: float = np.mean([res['oof_mv_recall'] for res in results_list])

    opt_f1_avg: float = np.mean([res['oof_opt_f1'] for res in results_list])
    opt_f1_std: float = np.std([res['oof_opt_f1'] for res in results_list])
    opt_precision_avg: float = np.mean([res['oof_opt_precision'] for res in results_list])
    opt_recall_avg: float = np.mean([res['oof_opt_recall'] for res in results_list])

    reopt_f1_avg: float = np.mean([res['oof_reopt_f1'] for res in results_list])
    reopt_f1_std: float = np.std([res['oof_reopt_f1'] for res in results_list])
    reopt_precision_avg: float = np.mean([res['oof_reopt_precision'] for res in results_list])
    reopt_recall_avg: float = np.mean([res['oof_reopt_recall'] for res in results_list])

    focal_loss_avg: float = np.mean([res['oof_focal_loss'] for res in results_list])

    mv_thresholds_avg: np.ndarray = np.array([res['oof_mv_thresholds'] for res in results_list]).mean(0)
    opt_threshold_avg: float = np.mean([res['oof_opt_threshold'] for res in results_list])
    reopt_threshold_avg: float = np.mean([res['oof_reopt_threshold'] for res in results_list])

    results = {
        'mv_f1_avg': mv_f1_avg,
        'mv_f1_std': mv_f1_std,
        'mv_precision_avg': mv_precision_avg,
        'mv_recall_avg': mv_recall_avg,

        'opt_f1_avg': opt_f1_avg,
        'opt_f1_std': opt_f1_std,
        'opt_precision_avg': opt_precision_avg,
        'opt_recall_avg': opt_recall_avg,

        'reopt_f1_avg': reopt_f1_avg,
        'reopt_f1_std': reopt_f1_std,
        'reopt_precision_avg': reopt_precision_avg,
        'reopt_recall_avg': reopt_recall_avg,

        'focal_loss_avg': focal_loss_avg,

        'mv_thresholds_avg': mv_thresholds_avg,
        'opt_threshold_avg': opt_threshold_avg,
        'reopt_threshold_avg': reopt_threshold_avg
    }
    return results
