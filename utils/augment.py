'''
Created on 29-June-2022

@author: Owaish
'''

from albumentations import *
import numpy as np
from PIL import Image
import torch
from albumentations.pytorch import ToTensorV2
from skimage import color
#imgagenet mean and std
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])



class HEDJitter(object):
    # Psuedo Code
    # 1. Change from RGB --> HED space
    # 2. Jitter channels
    # 3. HED--> RGB
    # 4. Blend white region from original image

    # theta determines the amount of jittering per channel in HED --> hematoxylin eosin DAB
    # theta = 0.02 is the defualt

    # 'th' is threshold for blending white background regions from original image. 'th' is in teh range [0,1]
    # default 'th' is 0.9 which corresponds to 0.9*255

    def __init__(self, theta=0.04):
        self.theta = theta
        self.th = 0.9
        self.cutoff_range = [0.15, 0.85]

    def __call__(self, img):
        patch_mean = np.mean(a=img) / 255.0
        if ((patch_mean <= self.cutoff_range[0]) or (patch_mean >= self.cutoff_range[1])):
            return (img)
        self.alpha = torch.distributions.uniform.Uniform(1 - self.theta, 1 + self.theta).sample(
            [1, 3])  # np.random.uniform(1 - self.theta, 1 + self.theta, (1, 3))
        self.beta = torch.distributions.uniform.Uniform(-self.theta, self.theta).sample(
            [1, 3])  # np.random.uniform(-self.theta, self.theta, (1, 3))
        # print(self.beta)
        img = np.array(img)
        s = color.rgb2hed(img)
        ns = self.alpha * s + self.beta  # perturbations on HED color space
        nimg = color.hed2rgb(ns)
        rsimg = np.clip(a=nimg, a_min=0.0, a_max=1.0)
        rsimg = (255 * rsimg).astype('uint8')
        return rsimg

class train_aug(object):
    '''
    Input: a numpy image (dim,dim,channel) and a mask (dim,dim)
    return a torch tensor for image and mask
    '''
    def __init__(self,hed=0.04,size = 512):
        
        self.HED = HEDJitter(hed)
        self.size = size
        self.color = Compose([
                        Resize(self.size, self.size),
                        #RGBShift(),
                        #OneOf([
                        Equalize(p=0.05),
                        HueSaturationValue(),
                        ColorJitter(),
                        Blur(),
                        RandomBrightnessContrast(),
                        ChannelShuffle(),
                        
                             ])
                        
        
        self.geometric = Compose([
                        Flip(),
                        #Sharpen(alpha=(0.25, 0.5), lightness=(0.5, 1.0)),
                        RandomRotate90(),
                        Transpose(),
                        #RandomScale([0.8, 1.2], 2),
                        #Rotate(limit=15, border_mode=cv2.BORDER_REFLECT),
                        #Blur(),
                        #GaussNoise()
                        ])
        
        self.norm = Compose([
                        Normalize(mean=mean, std=std),
                        ToTensorV2()])
        
        
    def __call__(self,image,mask):
        input_img = np.asarray(image)
        input_mask = np.asarray(mask)
        transformed = self.geometric(image=input_img,mask = input_mask)
        transformed = self.color(image = transformed['image'],mask = transformed['mask'])
        #transformed['image'] = self.HED(transformed['image'])
        transformed = self.norm(image = transformed['image'],mask = transformed['mask'])
        return transformed['image'], transformed['mask']
     
class val_aug(object):
    '''
    Input: a numpy image (dim,dim,channel) and a mask (dim,dim)
    return a torch tensor for image and mask
    '''
    def __init__(self,size = 512):
        self.size = size
        self.norm = Compose([
                Resize(self.size, self.size),
                Normalize(mean=mean, std=std),
                ToTensorV2()])
        
    def __call__(self,image,mask):
        input_img = np.asarray(image)
        input_mask = np.asarray(mask)
        transformed = self.norm(image = input_img,mask = input_mask)
        return transformed['image'], transformed['mask']




