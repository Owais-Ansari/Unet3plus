import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from soft_skeleton import soft_skel

ALPHA = 0.5
BETA = (1-ALPHA)
GAMMA = 1

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
    
     
def poly_cross_entropy1(logits,labels, epsilon = 1.0, num_classes=2, ignore_label=-1):
    '''
    Author:Owaish
    logits: Prediction after linear layer    
    labels: GT mask
    num_classes:num of classes
    ignore_label: label of ignore class
    epsilon : Parameter to tune the polyloss
    '''
    ce = nn.CrossEntropyLoss(weight=None,label_smoothing=0,reduction ='none',ignore_index=ignore_label)(logits,labels)
    probs = torch.nn.functional.softmax(logits,dim=1)
    #one_hot_encodig
    Y = torch.zeros(list(out.shape), dtype = torch.uint8)
    for i in range(list(out.shape)[0]):
        for j in range(list(out.shape)[1]):
            Y[i][j][y[i]==j]=1 
    pt = torch.sum(Y.cuda()*probs,dim = 1)
    poly_loss =  ce + epsilon * (1-pt)
    return torch.mean(poly_loss)    





def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, output, target, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        target=target.cpu()
        output= output.cpu()
        # comment out if your model contains a sigmoid or equivalent activation layer
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output)
        #output = torch.argmax(output, dim=1)
        # flatten label and prediction tensors
        output = output.view(-1)
        target = target.view(-1)
        # True Positives, False Positives & False Negatives
        TP = (output * target).sum()
        FP = ((1 - target) * output).sum()
        FN = (target * (1 - output)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky





class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super().__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self,y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]
    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]
    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred)[:,1:,...])
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true[:,1:,...]) + torch.sum(y_pred[:,1:,...]) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1.):
        super().__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self,y_true, y_pred):
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        y_true= y_true.float()
        y_pred= y_pred.float()
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice
    
    
    
    
