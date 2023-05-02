'''
Created on 26-April-2022
@author: owaish

'''
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

import argparse
import os,shutil,random,time,json
import numpy as np
import torch
import torch.nn as nn
#import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
##=================================================================================================================================
from torchsummary import summary
from utils.Metrics import SegmentationMetrics
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
from tqdm.autonotebook import tqdm

from timm.optim import AdamP,AdamW
from scipy.io import savemat
import torch.optim as optim
from dataloader import get_dataset,ProstateGleasonDataset
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
#from torch_ema import ExponentialMovingAverage
from utils.scheduler import TransformerLRScheduler
from numpy import record
from utils.losses import poly_cross_entropy1,DiceLoss
from utils.eval import pixel_acc
from sklearn.metrics import balanced_accuracy_score

from utils import metric
from utils.augment import train_aug, val_aug
#from DiceLoss import DiceLoss

import segmentation_models_pytorch as smp
from utils.models import Unet3plus
import config

# #===================================Arguments-start=======================================================================
parser = argparse.ArgumentParser(description = 'Segmentation Training')
# Datasets
parser.add_argument('-j', '--workers', default=config.num_workers, type=int, metavar='N',
                    help = 'number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default = config.epochs, type = int, metavar = 'N',
                    help = 'number of total epochs to run')
#parser.add_argument('--size', type=int, default=512, help='Input Image Size to Model.')
#
# # Organ dataset specific

parser.add_argument('--train-batch', default = config.train_batch, type = int, metavar = 'N',
                    help='train batch size')
# We keep train and test batch size as same
parser.add_argument('--lr', '--learning-rate', default = config.lr, type=float,
                    metavar = 'LR', help = 'initial learning rate')

parser.add_argument('--weight-decay', '--wd', default = config.weight_decay, type=float,
                    metavar = 'W', help = 'weight decay (default: 5e-4)')
#Checkpoints
parser.add_argument('-c', '--checkpoint', default=config.checkpoint, type=str, metavar='PATH',
                    help='checkpoint dir name (default: checkpoint)')

parser.add_argument('--resume', default = config.resume, type=str, metavar='PATH',
                    help = 'path to latest checkpoint (default: none)')

parser.add_argument('--seed', type=int,  default = config.seed, help = 'seed for reproducibility')

parser.add_argument('--gpu', type = int,  default = config.gpu)                    
parser.add_argument('--ignore_label', type=int,  default = config.ignore_label)                      
#parser.add_argument('--root_dir', type=str, metavar='', default='/mnt/store02/Beagle_dog_merck/10X_patches', help='Data Path')
parser.add_argument('--size', type=int, default = config.size, help='Input Image Size to Model.')


#parser.add_argument('--ema', default = None, type = float,
#                    help = 'Exponential moving average (in practise ema=0.95) helps in faster convergence. Large ema means less sensitive to current batch and vice-versa')
parser.add_argument('--clip', default = config.clip, type = float,
                    help = 'Visualize the range of gradient first the set the cut-off limit.This method only clip the norm/magnitude only. Gradient descent will still be in the same direction') 
parser.add_argument('--accum_iter', default=config.accum_iter, type=int,
                    help = 'if accumalation factor==1, that means no gradient accumalation') 

parser.add_argument('--cls_wghts', default=False, type=bool,
                    help='')
parser.add_argument('--label_sm', default= config.label_sm, type=float,
                    help='Label Smoothening to reduce the overconfidence level of prediction')
#parser.add_argument('--kl_div', type=bool, default=False)
#Device options
parser.add_argument('--freeze_backbone', default= config.freeze_backbone, type=bool,help='')
parser.add_argument('--ekd', default= False, type=bool,help='')
parser.add_argument('--num_classes', type=int,  default = config.num_classes)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
state= {}


num_workers = config.num_workers
epochs = config.epochs
train_batch = config.train_batch
lr = config.lr
weight_decay = config.weight_decay
checkpoint = config.checkpoint
resume = config.resume
gpu = config.gpu
seed = config.seed
clip = config.clip
size = config.size
ignore_label = config.ignore_label
accum_iter= config.accum_iter
label_sm = config.label_sm
freeze_backbone = config.freeze_backbone
num_classes  = config.num_classes
train_image_path = config.train_image_path
train_mask_path = config.train_mask_path
validation_image_path = config.validation_image_path
validation_mask_path = config.validation_mask_path


#===================================Arguments-end=======================================================================
# Use CUDA
use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
#torch.cuda.set_device(args.gpu)

#Random seed
if seed:
    random.seed(seed)
    torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)


def main():
    best_dice = 0
    #class_wise_dice = []
    start_epoch = 0
    #Create dataset

    #Loading dataset
    training_dataset = ProstateGleasonDataset(train_image_path, train_mask_path, ignore_label = num_classes, tfms = train_aug)
    validation_dataset = ProstateGleasonDataset(validation_image_path, validation_mask_path, ignore_label = num_classes, tfms = val_aug)

    trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=train_batch,  shuffle = True,  num_workers=num_workers)
    valloader   =   torch.utils.data.DataLoader(validation_dataset, batch_size=train_batch, shuffle=False, num_workers=num_workers)    
  
    print("")
    print("")
    
    device = torch.device('cuda:'+ str(gpu))
    model = Unet3plus(n_class = num_classes).cuda(device=device)

#=============================================================Freezing backbone except head ==================================================================================
#for name, param in model.segformer.encoder
    if freeze_backbone:
        for name, param in model.named_parameters():
        #list the names of parameters/layers which you want to train
            if 'econv' in name:
                param.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=lr )
    else:
        #2.Optimizer 
        optimizer = optim.AdamW(model.parameters(), lr = lr)

    model = model.to(device)
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    #Training parameters
    
    #1. Loss function
    criterion = {}
    criterion['ce'] = nn.CrossEntropyLoss(ignore_index = ignore_label,label_smoothing = label_sm)
    criterion['dice'] = DiceLoss(n_classes=num_classes)
    #Resume
    title = config.checkpoint
    checkpoint_dir = os.path.join('checkpoints', title)
    if not os.path.isdir(checkpoint_dir):
        mkdir_p(checkpoint_dir)
    print('Saving training files and checkpoints to {}'.format(checkpoint_dir)) 

    if resume:
        #Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resume)
        best_dice = checkpoint['best_dice']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(checkpoint_dir,'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(checkpoint_dir, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Train dice.', 'Valid Dice.'])
        #===================================scheduler==============================================================================
        #scheduler = LR_Scheduler(mode='cos', base_lr=args.lr, num_epochs=args.epochs, iters_per_epoch=len(trainloader))
    total_steps = int((epochs) * len(trainloader))
    resume_step = int(start_epoch * len(trainloader))
    warmup_steps = int(total_steps / 10)
    scheduler = TransformerLRScheduler(optimizer, init_lr=1e-4, peak_lr=lr, final_lr=1e-6, final_lr_scale=0.05,
                                   warmup_steps=warmup_steps, decay_steps=total_steps - warmup_steps)
    

    writer = SummaryWriter(log_dir=checkpoint_dir)
    with open(os.path.join(checkpoint_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent = 2)
    #Train and val
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, epochs):
        
        
        train_loss, train_acc, train_dice = train(trainloader, model, scaler, criterion, optimizer, scheduler, epoch, use_cuda, writer,checkpoint_dir = checkpoint_dir)
        test_loss, test_acc, test_dice, cm = test(valloader, model, criterion, epoch, use_cuda, num_classes = num_classes,conf_mat = True, checkpoint_dir = checkpoint_dir)
       

        writer.add_scalar('train/loss',train_loss, (epoch + 1))
        writer.add_scalar('train/acc', train_acc, (epoch + 1))
        writer.add_scalar('train/dice',train_dice, (epoch + 1))
        
        writer.add_scalar('val/loss', test_loss, (epoch + 1))
        writer.add_scalar('val/acc', test_acc, (epoch + 1))
        writer.add_scalar('val/dice', test_dice, (epoch + 1))
               
        
        
        #append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, train_dice,test_dice])
        
        
        #save model 
        if test_dice > best_dice:
            best_dice = max(test_dice, best_dice)
            #np.savetxt(os.path.join(checkpoint_dir,str(epoch+1)+'Best_conf_mat.csv'), cm, delimiter = ",")
            save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict':model.state_dict(), #to  avoid adding module in the keys name (model.state_dict() replace by model.module.state_dict())
                        'acc': test_acc,
                        #'bal_acc': test_bal_dice,
                        'best_dice': best_dice,
                        'optimizer' : optimizer.state_dict(),
                    }, test_dice > best_dice, checkpoint=checkpoint_dir)
            print("checkpoint is saved")
        print("---------------------------------------------------------------------------------------------------------------------")
        print("Saving dice score file")
        
    logger.close()
    print('Best dice:')
    print(best_dice)

def train(trainloader, model, scaler, criterion, optimizer, scheduler, epoch, use_cuda, writer,checkpoint_dir=''):
    #switch to train mode
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    dce = AverageMeter()
    end = time.time()
    
    img_path = []
    pred_name =[] 
    gt_name = []
    class_wise_dice = []
    progress_bar = tqdm(trainloader)
    
    for batch_idx, data in enumerate(progress_bar):

        inputs = data['image']
        targets = data['label']
        inputs, targets = inputs.cuda(torch.device('cuda:'+ str(gpu))), targets.cuda(torch.device('cuda:'+ str(gpu)))
        inputs, targets, paths = inputs, targets, data['path']       
        inputs, targets  = inputs.type('torch.cuda.FloatTensor'), targets
        
            
        #compute output (using mixed precision)
        with torch.set_grad_enabled(True):
        #optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                ce_loss = criterion['ce'](outputs,targets.long()).mean()
                #dice_loss = criterion['dice'](outputs,targets.long())
                loss =  1*ce_loss #+ 0*dice_loss
            
        dice_all, mean_dice = metric.seg_metric(outputs,targets,paths)
        dice = mean_dice
        # measure accuracy and record loss5xWSI
        acc = pixel_acc(outputs.data, targets.data)
        losses.update(loss.data, inputs.size(0))
        top1.update(acc, inputs.size(0))
        dce.update(dice.item(), inputs.size(0))
#================================================ Dump misclassified paths (examples which are hard to learn) ===========================================================
        for idp, pth in enumerate(list(dice_all.keys())):
            img_path.append(pth)
            class_wise_dice.append(list(dice_all.values())[idp])
#============================================== EMA+Gradient accumulation ==========================================================================================
        #Gradient accumatlation if  accum_iter > 1
        loss = loss / accum_iter
        #Compute gradient and do backpropagation step
        scaler.scale(loss).backward()
        #Weights update
        if ((batch_idx + 1) % accum_iter == 0)                                                                                                                                                                             or (batch_idx + 1 == len(trainloader)):
            #Gradient clipping prevtrain_diceenting grad exploding
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip) # check it first or keep it less than 0.999
            #Updating the optimizer state after back propagation
            #optimizer.step()
            scaler.step(optimizer)
            # Updates the scale (grad-scale) for next iteration 
            scaler.update()
            writer.add_scalar('train/loss_iter', loss.data, (epoch + 1)*len(progress_bar)+batch_idx)
            #accessing lr from optimizer state    
        
        state['lr'] =  scheduler.optimizer.param_groups[0]['lr']
            #update scheduler
            #scheduler.step()            
        writer.add_scalar('lr', state['lr'], (epoch + 1) * len(progress_bar) + batch_idx)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #plot progress
        progress_bar.set_description('(Epoch {epoch} | lr {lr} | {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | dice: {dice: .4f} '.format(
                    epoch=epoch + 1,
                    lr=state['lr'],
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt = batch_time.avg,
                    loss = losses.avg,
                    top1 = top1.avg,
                    dice = dce.avg
                    ))
    model.eval()
 
    savemat(checkpoint_dir + 'train_dice_score.mat', mdict = {'imgs_path':img_path,  'class_wise_dice': class_wise_dice})
    return losses.avg, top1.avg, dce.avg
    #-------------------------------------------------------------------------------------------------------------------------------


def test(testloader, model, criterion, epoch, use_cuda=True, writer=None, num_classes=2, conf_mat=False,checkpoint_dir=''):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    dce = AverageMeter()
    pre = AverageMeter()
    rec = AverageMeter()
    conf_matrix = np.zeros((num_classes, num_classes))
    class_wise_dice = []
    img_path = []
    #switch to evaluate mode
    model.eval()

    end = time.time()
    progress_bar = tqdm(testloader)
    for batch_idx, data in enumerate(progress_bar):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
           
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            inputs = data['image']
            targets = data['label']
            paths = data['path']  
             
            inputs, targets  = inputs.cuda(torch.device('cuda:'+ str(gpu))), targets.cuda(torch.device('cuda:'+ str(gpu))) 
            
        with torch.no_grad():
        #optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                ce_loss = criterion['ce'](outputs,targets.long()).mean()
                #dice_loss = criterion['dice'](outputs,targets.long())
                loss =  1*ce_loss #+ 0*dice_loss

        dice_all, mean_dice = metric.seg_metric(outputs,targets,paths)
        dice = mean_dice
        acc = pixel_acc(outputs.data, targets.data)
        
        for idp, pth in enumerate(list(dice_all.keys())):
            img_path.append(pth)
            class_wise_dice.append(list(dice_all.values())[idp])

      
        if conf_mat:
            y = targets.detach().cpu().numpy().flatten().astype(np.uint8)
            y_p = torch.argmax(outputs, dim=1).detach().cpu().numpy().flatten().astype(np.uint8)
            y_true = list(y)
            y_pred = list(y_p)
            conf_matrix += confusion_matrix(y_true, y_pred, labels=range(num_classes)) 
            
        # measure accuracy and record loss
        losses.update(loss.data, inputs.size(0))
        top1.update(acc, inputs.size(0))
        
        dce.update(dice.item(), inputs.size(0))

        #measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #plot progress
        progress_bar.set_description('| Validation Loss: {loss:.4f} | Validation top1: {top1: .4f} | dice {dice1: .4f}'.format(
            loss = losses.avg,
            top1 = top1.avg,
            dice1 = dce.avg
        ))
    savemat(checkpoint_dir + 'val_dice_score.mat', mdict = {'imgs_path':img_path,'class_wise_dice': class_wise_dice})

    return (losses.avg, top1.avg, dce.avg, conf_matrix)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    #filepath = os.path.join(checkpoint, str(state['epoch'])+'_'+filename)
    filepath = os.path.join(checkpoint, str(filename))
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        
        
if __name__ == '__main__':
    main()
            