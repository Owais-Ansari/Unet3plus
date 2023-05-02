

import numpy as np
import os,argparse,glob,random
from torch.utils.data import Dataset
from PIL import Image
import cv2

    
class get_dataset(Dataset):
    def __init__(self, image_path, mask_path, ignore_label=255,tfms=None):
        self.imgs_path = [img_p for img_p in glob.glob(image_path + '/*.png') if not 'Image' in img_p]
        #self.fnames = [img_p for img_p in self.fnames if not 'TCGA-CH-5763-01Z-00-DX1.7d4eff47-8d99-41d4-87f0-163b2cb034bf' in img_p] 
        #self.fnames = [img_p for img_p in self.fnames if not 'TCGA-EJ-5532-01Z-00-DX1.07548d1f-867f-4bca-a9f3-aaed043b8753' in img_p]
        #self.fnames = [img_p for img_p in self.fnames if not 'TCGA-YL-A9WY-01Z-00-DX1.16415C29-1D79-4560-8F49-633F37F2804E' in img_p] 
        len(self.imgs_path)
        self.tfms = tfms
        self.mask_path = mask_path
        self.ignore_label = ignore_label
        
    def __len__(self):
        return len(self.imgs_path)
    
    def __getitem__(self, idx):
        
        img_path = self.imgs_path[idx]
        mskname = os.path.basename(img_path)
        #try:
        img  = np.asarray(Image.open(img_path),dtype=np.uint8)
        mask = np.asarray(Image.open(self.mask_path+mskname).convert('L'), dtype = np.uint8)
        mask = mask.copy()
        

        mask[mask == 255] = self.ignore_label

            
        if self.tfms is not None:
            augmented = self.tfms()
            #augmented = self.tfms(image=img, mask=mask)
            img,mask = augmented(img,mask)
        return {'image':img,'label':mask,'path':img_path}

