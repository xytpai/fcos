import torch
import torch.nn as nn



def smooth_l1_loss(pred, target, beta=0.11):
    x = (pred - target).abs()
    l1 = x - 0.5 * beta
    l2 = 0.5 * x ** 2 / beta
    return torch.where(x >= beta, l1, l2)
