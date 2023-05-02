from __future__ import print_function, absolute_import
import torch
__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# import numpy as np
# import sklearn.metrics as metrics
# out = torch.randint(0,2,(6,4,512,512))
# gt = torch.randint(0,4,(6,512,512))

#print(metrics.confusion_matrix(gt.reshape(-1),out.reshape(-1), labels =[0,1,2]))



def pixel_acc(pred, label):
    _, preds = torch.max(pred, dim=1)
    #print(torch.unique(pred[:,0,:,:]))
    valid = (label >= 0).long()
    acc_sum =torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

#print(pixel_acc(out,gt))


#_, predicted = out.max(1)