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
    def __init__(self,hed=0.04):
        
        self.HED = HEDJitter(hed)
        self.color = Compose([
                        Resize(512, 512),
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
    def __init__(self):
        self.norm = Compose([
                Resize(512, 512),
                Normalize(mean=mean, std=std),
                ToTensorV2()])
    def __call__(self,img,mask):
        input_img = np.asarray(img)
        input_mask = np.asarray(mask)
        transformed = self.norm(image = input_img,mask = input_mask)
        return transformed['image'], transformed['mask']




# class aug(object):
#     def __init__(self):
#         self.morphology = A.Compose([
#             A.OneOf([
#                 A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0.1),
#                 A.Sharpen(alpha=(0.25, 0.5), lightness=(0.5, 1.0)),
#             ], p=0.75),
#             A.RandomScale([0.8, 1.2], 2),
#             A.transforms.GaussNoise(var_limit=[0.0, 0.1])
#         ])
#
#numpy
#
#         self.bc = A.Compose([A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5)])
#         self.hed_only = HEDJitter(theta)
#
#     def __call__(self, input):
#
#         input = np.asarray(input)
#         transformed = self.morphology(image=input)
#
#         # random elastic deformation
#         alpha = np.random.randint(low=80, high=120)
#         sigma = np.random.randint(low=9, high=11)
#         transformed['image'] = A.elastic_transform(transformed['image'], alpha=alpha,
#                                           sigma=sigma, alpha_affine=50, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT)
#
#         # color aug
#         transformed = self.bc(image=transformed['image'])
#         transformed['image'] = self.hed_only(transformed['image'])
#
#         transformed['image'] = Image.fromarray(transformed['image'].astype('uint8'), 'RGB')
#
#         return transformed['image']
    
    