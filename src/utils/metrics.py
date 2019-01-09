from sklearn.metrics import f1_score


def f1(outputs, targets):
    outputs = outputs.cpu().numpy().reshape(-1,)
    targets = targets.cpu().numpy().reshape(-1,)
    return f1_score(outputs, targets)
