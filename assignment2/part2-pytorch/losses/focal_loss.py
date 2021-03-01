import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = list()
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    n_class = 10
    for samples_per_cls in cls_num_list:
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * n_class
        per_cls_weights.append(weights)
    per_cls_weights = torch.from_numpy(np.array(per_cls_weights))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        target = F.one_hot(target, num_classes=10).type_as(input)
        loss = F.binary_cross_entropy_with_logits(
            input=input, target=target, reduction="none"
        )

        loss *= torch.exp(
            -self.gamma * target * input - self.gamma * torch.log(1 + torch.exp(-1.0 * input))
        )

        wt_loss = loss * self.weight

        loss = torch.sum(wt_loss) / torch.sum(target)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
