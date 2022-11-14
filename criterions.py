import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight=False):
        super().__init__()
        if reweight:
            class_idx = np.array(cls_num_list) != 0
            effective_num = 1.0 - np.power(0.999, cls_num_list)
            per_cls_weights = np.zeros_like(effective_num)
            per_cls_weights[class_idx] = (1.0 - 0.999) / np.array(effective_num)[class_idx]
            per_cls_weights[class_idx] = per_cls_weights[class_idx] / np.sum(per_cls_weights[class_idx]) * class_idx.sum()
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)

        return self

    def forward(self, output_logits, target):  # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)