from tqdm import tqdm
from sklearn.metrics import f1_score


def f1(outputs, targets, threshold=0.5):
    outputs = outputs.cpu().numpy().reshape(-1,)
    targets = targets.cpu().numpy().reshape(-1,)
    return f1_score(outputs, targets > threshold)


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
