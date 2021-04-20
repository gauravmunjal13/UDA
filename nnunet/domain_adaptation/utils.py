import numpy as np
import torch
import torch.nn as nn

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

def lr_poly(base_lr, i_iter, max_iters, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(i_iter) / max_iters) ** power)

def _adjust_learning_rate(optimizer, lr_D, i_iter, max_iters, power): 
    lr = lr_poly(lr_D, i_iter, max_iters, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_discriminator(optimizer, lr_D, i_iter, max_iters, power):
    _adjust_learning_rate(optimizer, lr_D, i_iter, max_iters, power)