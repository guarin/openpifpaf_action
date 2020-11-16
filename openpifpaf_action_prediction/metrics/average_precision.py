import torch


def voc_ap(
    scores,
    target,
    column=None,
    threshold=0.5,
    eps=torch.finfo(torch.float32).eps,
    return_all=False,
):
    """Average precision from Pascal VOC 2010/2012.

    Adapted from https://github.com/s-gupta/v-coco.
    """
    # select column from batch if passed as argument
    if column is not None:
        scores = scores[:, column]
        target = target[:, column]

    # sort in decreasing order
    index = scores.argsort(descending=True)
    scores = scores[index]
    target = target[index]

    # output predictions
    output = (scores >= threshold).int()

    # cumulative true positives, false positives and false negatives
    tp = torch.cumsum((output == 1) & (target == 1), 0).float()
    fp = torch.cumsum((output == 1) & (target == 0), 0).float()

    precision = tp / (tp + fp + eps)
    recall = tp / (torch.sum(target) + eps)

    # add initial and final values
    one = scores.new_ones(1)
    zero = scores.new_zeros(1)
    precision = torch.cat([one, precision, zero])
    recall = torch.cat([zero, recall, one])

    # precision[i] = max(precision[j]) for j >= i
    mask = torch.triu(precision.new_ones(precision.shape[0], precision.shape[0]))
    precision = (precision * mask).max(1)[0]

    # calculate area under the curve
    auc = torch.sum((recall[1:] - recall[:-1]) * precision[1:])

    if return_all:
        return auc, recall, precision
    return auc
