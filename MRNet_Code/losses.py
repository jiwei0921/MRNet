# -*- coding: utf-8 -*-

import torch.nn.functional as F
from torch import nn
import torch


class BCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.bce_loss(output, target)

class CELoss(nn.Module):
    def __init__(self, **kwargs):
        super(CELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.ce_loss(output, target)

class CELoss2d(nn.Module):
    def __init__(self, **kwargs):
        super(CELoss2d, self).__init__()

    def cross_entropy2d(self, input, target, weight=None, size_average=True):
        n, c, h, w = input.size()

        input = input.transpose(1, 2).transpose(2, 3).contiguous()
        input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        input = input.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = F.cross_entropy(input, target, weight=weight, size_average=False)
        if size_average:
            loss /= mask.data.sum()
        return loss

    def forward(self, output, target, weight=None, size_average=True):
        return self.cross_entropy2d(output, target,weight, size_average)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))
        return loss_contrastive


"""Create loss"""
__factory = {
    'cross_entropy': CELoss,
    'cross_entropy2d': CELoss2d,
    'BCE_logit': BCELoss,
    'ContrastiveLoss':ContrastiveLoss,
    }


def get_names():
    return __factory.keys()


def init_loss(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown loss: {}".format(name))
    return __factory[name](**kwargs)



if __name__ == '__main__':
    pass


