import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothF1Loss(nn.Module):
    def __init__(self, logit=True, epsilon=1e-10):
        super(SmoothF1Loss, self).__init__()
        self.logit = logit
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        if self.logit:
            outputs = torch.sigmoid(outputs)
        true_positive = outputs * targets
        precision = true_positive / (outputs + self.epsilon)
        recall = true_positive / (targets + self.epsilon)
        smooth_f1_score = 2 * precision * recall / (precision + recall + self.epsilon)
        loss = torch.mean(1 - smooth_f1_score, 0)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, logit=True, gamma=2.0, alpha=0.25, epsilon=1e-10):
        super(FocalLoss, self).__init__()
        self.logit = logit
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        if self.logit:
            outputs = torch.sigmoid(outputs)
        outputs = torch.clamp(outputs, min=self.epsilon, max=1-self.epsilon)

        p_t = targets * outputs + (1 - targets) * (1 - outputs)
        alpha = torch.ones_like(outputs) * self.alpha
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)

        cross_entropy = F.binary_cross_entropy(outputs, targets)
        weight = alpha_t * torch.pow((1 - p_t), self.gamma)
        loss = weight * cross_entropy
        loss = torch.mean(loss, 0)
        return loss


class L2Regulaization(nn.Module):
    exclude_regex = re.compile(r'bn|bias|activation')

    def __init__(self, reg_lambda):
        super(L2Regulaization, self).__init__()
        self.reg_lambda = reg_lambda

    def forward(self, model: nn.Module):
        loss = 0
        for name, weights in model.named_parameters():
            if self.exclude_regex.search(name):
                continue
            loss += weights.norm(2)

        return loss * self.reg_lambda


class MultiTaskLoss(nn.Module):
    def __init__(self, losses, weights):
        super(MultiTaskLoss, self).__init__()
        assert len(losses) == len(weights)
        self.losses = losses
        self.weights = weights

    def forward(self, inputs, targets):
        assert len(inputs) == len(targets)
        assert len(inputs) == len(self.losses)
        total_loss = torch.FloatTensor(1)
        for x, y, loss, w in zip(inputs, targets, self.losses, self.weights):
            total_loss += w * loss(x, y)
        return total_loss


class MaskLoss(nn.Module):
    def __init__(self, criterion):
        super(MaskLoss, self).__init__()
        self.criterion = criterion

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        loss = torch.FloatTensor(1)
        masks = torch.sum(targets, (2, 3)).view(-1) > 0
        for y_pred, y_true, mask in zip(inputs, targets, masks):
            sample_loss = self.criterion(y_pred.unsqueeze(0), y_true.unsqueeze(0)) * mask.float()
            loss += sample_loss
        return loss / batch_size
