import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / (self.count+1e-10)

    def value(self):
        return self.val

    def average(self):
        return self.avg


class BinCounterMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, values):
        self.values = values
        self.hist_map = {}
        self.count = 0

        for val in self.values:
            self.hist_map[str(val)]=0

    def update(self, unique, counts):
        for ind, val in enumerate(unique):
            self.hist_map[str(val)]+=counts[ind]
        self.count+=np.sum(counts)

    def get_distribution(self):
        res = np.zeros(self.values.shape)
        for ind, val in enumerate(self.values):
            res[ind] = self.hist_map[str(val)]
        return res/self.count


def accuracy(trg, pred):
    _, preds = torch.max(pred, dim=1)
    valid = (trg >= 0)
    acc = 1.0 * torch.sum(valid * (preds == trg)) / (torch.sum(valid) + 1e-10)
    return acc.item(), torch.sum(valid).item()


def check_grad(params, clip_th, ignore_th):
    befgad = torch.nn.utils.clip_grad_norm(params, clip_th)
    return (not np.isfinite(befgad) or (befgad > ignore_th))


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %.8f\n" % (optimizer.param_groups[0]['lr'],))