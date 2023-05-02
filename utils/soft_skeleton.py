import torch
import torch.nn as nn
import torch.nn.functional as f


def soft_erode(img):
    if len(img.shape)==4:
        p1 = -f.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -f.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape)==5:
        p1 = -f.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -f.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -f.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape)==4:
        return f.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return f.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  f.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  f.relu(img-img1)
        skel  =  skel +  f.relu(delta-skel*delta)
    return skel