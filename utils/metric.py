import torch

# def IoU(output, target, epsilon=1e-2):
#     intersection = torch.sum(output * target, (1,2))
#     union = torch.sum(output + target, (1,2)) 
#     return (intersection+epsilon)/(union-intersection+epsilon)
#
#
# def DiceScore(output, target, epsilon=1e-2):
#     intersection = 2*torch.sum(output * target, (1,2)) + epsilon
#     union = torch.sum(output + target, (1,2)) + epsilon
#     return intersection/union

#============================image level metric ===========================================================
def DiceScore(output, target, epsilon=1e-3):
    intersection = 2*torch.sum(output * target) + epsilon
    union = torch.sum(output + target) + epsilon
    return intersection/union

def IoU(output, target, epsilon=1e-2):
    intersection = torch.sum(output * target)
    union = torch.sum(output + target) 
    return (intersection+epsilon)/(union-intersection+epsilon)

def Precision_seg(output, target, epsilon=1e-2):
    intersection = torch.sum(output * target) # true positive
    fp = torch.sum(target - output * target) #positive prediction - TP
    return (intersection+epsilon)/(fp+intersection+epsilon)

def Recall_seg(output, target, epsilon=1e-2):
    intersection = torch.sum(output * target)
    fn = torch.sum(output - output * target)
    return (intersection+epsilon)/(fn+intersection+epsilon)
#==========================================================================================================
def dice_coeff(y_pred,y_true):
    """[function to compute dice loss]
    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]
    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    #y_pred = torch.argmax(y_pred,dim=1)
    intersection = torch.sum((y_true * y_pred)[:,1:,...])
    coeff = (2. * intersection + smooth) / (torch.sum(y_true[:,1:,...]) + torch.sum(y_pred[:,1:,...]) + smooth)
    return coeff

import numpy as np

def seg_metric(out,target,path):
    num_classes =  out.shape[1]
    DICE_all  = {}
    dice_temp = {}
    for i in range(out.shape[0]): # loop across batch size
        target_class = target[i, :, :] # multilabel mask
        ignore = (target_class == num_classes) #num_classes value is same as ignore label index
        #print(torch.unique(target_class), torch.unique(torch.argmax(out[i, :, :, :], dim=0)))
        for j in range(num_classes):
            #across classes
            binary_out =  torch.zeros_like(target_class)  
            binary_y = torch.zeros_like(target_class) 
            #WXH
            binary_y[target_class==j] = 1
            binary_out[torch.argmax(out[i, :, :, :],   dim=0).detach().cpu()==j] = 1

            binary_y[ignore == True] = 0
            binary_out[ignore==True] = 0

            dice = DiceScore(binary_out,binary_y)
            dice_temp[j]= np.float(dice.type('torch.FloatTensor'))
        DICE_all[str(path[i])] = dice_temp
    batch_dice = 0
    for i in range(0,len(DICE_all)):
        img_mean_dice = np.mean([val for val in DICE_all[str(path[i])].values()])
        batch_dice = img_mean_dice + batch_dice 
    batch_dice = batch_dice / len(DICE_all)
    return DICE_all,batch_dice
            

# def seg_metric(out,target,path):
#     num_classes =  out.shape[1]
#     DICE_all  = {}
#     dice_temp = {}
#     tmp = 0
#     for i in range(0,out.shape[0]): # loop across batch size
#         target_class = target[i, :, :] # multilabel mask
#         ignore = (target_class == num_classes) #num_classes value is same as ignore label index
#         for j in range(num_classes):
#              #across classes
#             binary_out =  torch.zeros_like(target_class)  
#             binary_y = torch.zeros_like(target_class) 
#                                        #WXH
#             binary_y[target_class==j] = 1
#             binary_out[torch.argmax(out[i, :, :, :],dim=0).detach().cpu()==j] = 1
#             binary_y[ignore == True] = 0
#             binary_out[ignore==True] = 0
#             dice = DiceScore(binary_out,binary_y)
#             dice_temp[j]= np.float(dice.type('torch.FloatTensor'))
#     if (i+1)%4==0 or i==0:
#         DICE_all[str(path[tmp])] = dice_temp
#         tmp = tmp + 1
#     batch_dice = 0
#     for i in range(0,len(DICE_all)):
#         img_mean_dice = np.mean([val for val in DICE_all[str(path[i])].values()])
#         batch_dice = img_mean_dice + batch_dice 
#     batch_dice = batch_dice / len(DICE_all)
#     return DICE_all,batch_dice
            

            
            
            
            
        
        
        
        
    #
    # for i in range(0,num_classes,1):
    #     binary_out = torch.zeros_like(out.cpu())
    #     binary_out[out.cpu()==i] = 1
    #     binary_y = torch.zeros_like(target.squeeze(1))
    #     binary_y[target==i] = 1
    #     element_dice = DiceScore(binary_out, binary_y)
    #     element_iou = IoU(binary_out, binary_y)
    #     Temp_Class_1[i].append(list(element_dice.numpy()))
    #     Temp_Class_2[i].append(list(element_iou.numpy()))
    #     if i!=0:
    #         Temp_1.append(list(element_dice.numpy()))
    #         Temp_2.append(list(element_iou.numpy()))
    # return 














