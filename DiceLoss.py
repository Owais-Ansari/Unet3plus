import numpy as np
import torch
import torch.nn as nn

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
    
def poly_cross_entropy1(logits,labels, epsilon = 1.0, num_classes=4, ignore_label=255):
    '''
    Author: Owaish
    TODO: weights for classes
    '''
    probs = torch.nn.functional.softmax(logits,dim=1)
    #one_hot_encodig
    Y = torch.zeros(list(out.shape),dtype=torch.uint8)
    for i in range(list(out.shape)[0]):
        #weights_class = []
        for j in range(list(out.shape)[1]):
            Y[i][j][y[i]==j]=1 
            #weights_class.append(torch.sum(Y[i][j][y[i]==j]))
    pt = torch.sum(Y.cuda()*probs,dim = 1)
    ce = nn.CrossEntropyLoss(weight=None,label_smoothing=0,reduction ='none',ignore_index=ignore_label)(logits,labels)
    poly_loss = ce + epsilon*(1-pt)
    return torch.mean(poly_loss)