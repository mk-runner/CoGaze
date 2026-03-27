"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples"
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BalancedClassLoss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes=4, loss_type='focal', gamma=0.2, beta=0.9999):
        """

        Args:
              samples_per_cls: A python list of size [no_of_classes].
              no_of_classes: total number of classes. int
              loss_type: string. One of "sigmoid", "focal".
              beta: float. Hyperparameter for Class balanced loss.
              gamma: float. Hyperparameter for Focal loss.
        """
        super().__init__()
        self.loss_type = loss_type
        self.gamma = gamma
        self.no_of_classes = no_of_classes
        # precompute effective number weights
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, labels):
        """

        Args:
            logits (Tensor): shape [batch_size, num_classes]
            labels (Tensor): shape [batch_size] with integer class labels

        Returns:
            loss (Tensor): scalar loss value
        """
        device = logits.device
        labels_one_hot = F.one_hot(labels, self.no_of_classes).float().to(device)
        weights = self.class_weights.to(device)

        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1).unsqueeze(1).repeat(1, self.no_of_classes)
        if self.loss_type == 'focal':
            loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == 'sigmoid':
            loss = F.binary_cross_entropy_with_logits(logits, labels_one_hot, weight=weights)
        else:
            raise ValueError(f'Unsupported loss_type: {self.loss_type}')
        return loss


class MultiLabelBalancedClassLoss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes=4, loss_type='focal', gamma=0.2, beta=0.9999):
        """
        Args:
            samples_per_cls: list, 每个类别的样本数
            no_of_classes: 类别总数
            loss_type: "sigmoid" or "focal"
            beta: class balanced loss 的超参数
            gamma: focal loss 的超参数
        """
        super().__init__()
        self.loss_type = loss_type
        self.gamma = gamma
        self.no_of_classes = no_of_classes

        # 计算 Class-Balanced 权重
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, labels):
        """
        Args:
            logits: [batch_size, num_classes]
            labels: [batch_size, num_classes], 多标签 0/1
        """
        device = logits.device
        labels = labels.float().to(device)  # 保证是 float tensor
        weights = self.class_weights.to(device)

        # 每个类别独立加权
        if self.loss_type == 'sigmoid':
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        elif self.loss_type == 'focal':
            # multilabel focal loss
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss_item = ((1 - pt) ** self.gamma) * bce_loss
            loss = (focal_loss_item * weights).mean()
        elif self.loss_type == 'cross-entropy':
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            raise ValueError(f'Unsupported loss_type: {self.loss_type}')

        return loss


class MultiLabelDiseasesBalancedClassLoss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes=14, loss_type='focal', gamma=0.2, beta=0.9999):
        """
        Args:
            samples_per_cls: list[int], 每个类别的样本数
            no_of_classes: 类别总数
            loss_type: "sigmoid" or "focal"
            beta: class balanced loss 的超参数
            gamma: focal loss 的超参数
        """
        super().__init__()
        self.loss_type = loss_type
        self.gamma = gamma
        self.no_of_classes = no_of_classes

        # 计算 Class-Balanced 权重
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / (np.array(effective_num) + 1e-8)
        weights = weights / np.sum(weights) * no_of_classes
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits, labels):
        """
        Args:
            logits: [B, num_classes] (raw outputs, no sigmoid)
            labels: [B, num_classes] (0/1 targets)
        """
        device = logits.device
        labels = labels.float().to(device)       # [B, C]
        weights = self.class_weights.to(device)  # [C]
        weights = weights.unsqueeze(0)           # [1, C] → 可广播到 [B, C]

        # 每个类别独立加权
        if self.loss_type == 'sigmoid':
            # Weighted BCE
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, weight=weights, reduction='mean'
            )

        elif self.loss_type == 'focal':
            # multilabel focal loss
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            pt = torch.exp(-bce_loss).clamp(min=1e-8, max=1.0)
            focal_loss_item = ((1 - pt) ** self.gamma) * bce_loss
            loss = (focal_loss_item * weights).mean()
        else:
            raise ValueError(f'Unsupported loss_type: {self.loss_type}')

        return loss


class MultiBalancedClassLoss(nn.Module):
    def __init__(self, class_frequency, loss_type='focal', gamma=0.2, beta=0.9999):
        """

        Args:
              class_frequency: list of size [[no_of_classes], ..., [no_of_classes]].
              loss_type: string. One of "sigmoid", "focal", "softmax".
              beta: float. Hyperparameter for Class balanced loss.
              gamma: float. Hyperparameter for Focal loss.
        """
        super().__init__()
        self.balanced_class_func = []
        self.num_tasks = len(class_frequency)
        for class_freq in class_frequency:
            no_of_classes = len(class_freq)
            balanced_class_func = BalancedClassLoss(class_freq, no_of_classes, loss_type, gamma, beta)
            self.balanced_class_func.append(balanced_class_func)

    def forward(self, logits, labels):
        """

        Args:
            logits (List[Tensor]): List of logits tensors, each of shape [batch_size, C]
            labels (Tensor): shape [batch_size, C] with integer class labels

        Returns:
            loss (Tensor): scalar loss value
        """
        total_loss = 0

        for i in range(self.num_tasks):
            logit = logits[i]  # (B, C)
            target = labels[:, i]  # (B,)
            loss = self.balanced_class_func[i](logit, target)
            total_loss += loss
        return total_loss / self.num_tasks


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")  # log(pt), negative values

    # (1-pt)^gamma, where pt = sigmoid(logits)
    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                           torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss   # not have "-"
    # focal_loss = torch.sum(weighted_loss) / torch.sum(labels)   # only suitable for one-hot encoding
    focal_loss = torch.mean(weighted_loss)

    return focal_loss


# def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
#     """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
#
#     Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
#     where Loss is one of the standard losses used for Neural Networks.
#
#     Args:
#       labels: A int tensor of size [batch].
#       logits: A float tensor of size [batch, no_of_classes].
#       samples_per_cls: A python list of size [no_of_classes].
#       no_of_classes: total number of classes. int
#       loss_type: string. One of "sigmoid", "focal", "softmax".
#       beta: float. Hyperparameter for Class balanced loss.
#       gamma: float. Hyperparameter for Focal loss.
#
#     Returns:
#       cb_loss: A float tensor representing class balanced loss
#     """
#     effective_num = 1.0 - np.power(beta, samples_per_cls)
#     weights = (1.0 - beta) / np.array(effective_num)
#     weights = weights / np.sum(weights) * no_of_classes
#
#     labels_one_hot = F.one_hot(labels, no_of_classes).float()
#
#     weights = torch.tensor(weights).float()
#     weights = weights.unsqueeze(0)
#     weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
#     weights = weights.sum(1)
#     weights = weights.unsqueeze(1)
#     weights = weights.repeat(1, no_of_classes)
#
#     if loss_type == "focal":
#         cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
#     elif loss_type == "sigmoid":
#         cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weights=weights)
#     elif loss_type == "softmax":
#         pred = logits.softmax(dim=1)
#         cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
#     return cb_loss


# if __name__ == '__main__':
#     no_of_classes = 5
#     logits = torch.rand(10, no_of_classes).float()
#     labels = torch.randint(0, no_of_classes, size=(10,))
#     beta = 0.9999
#     gamma = 2.0
#     samples_per_cls = [2, 3, 1, 2, 2]
#     loss_type = "focal"
#     cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma)
#     print(cb_loss)
#
#
#     temp = BalancedClassLoss(samples_per_cls, no_of_classes, loss_type, gamma, beta)
#     print(temp(logits, labels))
