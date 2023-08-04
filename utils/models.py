import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import numpy as np

from timm import create_model



def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )   

def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


def conv2dTranspose(in_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2,stride=2))

def masking(in_channels):
    return nn.Sequential( 
                    nn.Dropout(p=0.5),
                    nn.Conv2d(in_channels, 2, 1),
                    nn.AdaptiveMaxPool2d(1),
                    nn.Sigmoid()
                            )

   
class UNet3plus(nn.Module):

    def __init__(self, num_classes, encoder = "pvt_v2_b2_li"):
        super().__init__()
                            
        enc =  timm.create_model(encoder, pretrained = True)

        self.econv0 = enc.patch_embed #64   ,256X256
        self.econv1 = enc.stages[0] #64    ,256X256
        self.econv2 = enc.stages[1]
        self.econv3 = enc.stages[2]
        self.econv4 = enc.stages[3]
        
        self.dconv4 =  double_conv(512, 256)  #,
        self.single4 = single_conv(768,256)
        
        self.dconv3 =  double_conv(256, 128)
        self.single3 = single_conv(320,256)
        
        self.dconv2 = double_conv(256, 64)
        self.dconv1 = double_conv(128, 32)
        #self.dconv0 = double_conv(64+32, 32)
        
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(32, num_classes, 1)
        
        
    def forward(self, x):
        #x 1024X1021       
        conv0 = self.econv0(x) # 512X512
        conv1 = self.econv1(conv0[0], feat_size = conv0[1])# 256X256
        conv2 = self.econv2(conv1[0], feat_size = conv1[1])
        conv3 = self.econv3(conv2[0], feat_size = conv2[1])# 64X64
        conv4 = self.econv4(conv3[0], feat_size = conv3[1])# 32X32
        
        
        dec4 = self.dconv4(conv4[0])
        dec4 = torch.cat([self.upsample(dec4), conv3[0], self.maxpool(conv2[0]), self.maxpool(self.maxpool(conv1[0]))], dim=1)# 64X64
        
        dec4 = self.single4(dec4)
        
        dec3 = self.dconv3(dec4)
        #dec3 = torch.cat([self.upsample(dec3), conv2[0]], dim=1)# 128X128
        dec3 = torch.cat([self.upsample(dec3), conv2[0],self.maxpool(conv1[0])], dim=1)# 128X128
        
        dec3 = self.single3(dec3)
        
        dec2 = self.dconv2(dec3)
        dec2 = torch.cat([self.upsample(dec2), conv1[0]], dim=1)# 256X256
        dec1 = self.dconv1(dec2)
        out = self.conv_last(dec1)

        return out
    
class Unet3plusGlcm(nn.Module):

    def __init__(self, num_classes, encoder = "pvt_v2_b2_li"):
        super().__init__()
                         
        enc =  timm.create_model(encoder, pretrained = True)
        
        self.econv0 = enc.patch_embed #64   ,256X256
        self.econv1 = enc.stages[0] #64    ,256X256
        self.econv2 = enc.stages[1]
        self.econv3 = enc.stages[2]
        self.econv4 = enc.stages[3]
        
        self.dconv4 =  double_conv(512, 256)  #,
        self.single4 = single_conv(768,256)
        
        self.dconv3 =  double_conv(256, 128)
        self.single3 = single_conv(832,256)
        
        self.dconv2 = double_conv(256, 64)
        self.dconv1 = double_conv(896, 32) #128
        #self.dconv0 = double_conv(64+32, 32)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(32, num_classes, 1)
        self.cls = masking(512)
        
    def dotProduct(self, seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final    
 
    def forward(self, x):
        #x 1024X1021       
        conv0 = self.econv0(x) # 512X512
        conv1 = self.econv1(conv0[0], feat_size = conv0[1])# 256X256
        conv2 = self.econv2(conv1[0], feat_size = conv1[1])
        conv3 = self.econv3(conv2[0], feat_size = conv2[1])# 64X64
        conv4 = self.econv4(conv3[0], feat_size = conv3[1])# 32X32
        
    #====================================================================================================================================
        cls_branch = self.cls(conv4[0]).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
        cls_branch_max = cls_branch.argmax(dim=1)
        cls_branch_max = cls_branch_max[:, np.newaxis].float()
        #=====================================================================================================================================
        
        
        dec4 = self.dconv4(conv4[0])
        dec4 = torch.cat([self.upsample(dec4),conv3[0],self.maxpool(conv2[0]),self.maxpool(self.maxpool(conv1[0]))],dim=1)# 64X64
        
        dec4 = self.single4(dec4)
        
        dec3 = self.dconv3(dec4)
        #dec3 = torch.cat([self.upsample(dec3), conv2[0]], dim=1)# 128X128
        dec3 = torch.cat([self.upsample(dec3), self.upsample(self.upsample(conv4[0])), conv2[0],self.maxpool(conv1[0])], dim=1)# 128X128
        
        dec3 = self.single3(dec3)
        
        dec2 = self.dconv2(dec3)
        dec2 = torch.cat([self.upsample(dec2),self.upsample(self.upsample(self.upsample(conv4[0]))), self.upsample(self.upsample(dec4)), conv1[0]], dim=1)# 256X256
        dec1 = self.dconv1(dec2)
        out  = self.conv_last(dec1)
        
        out = self.dotProduct(out, cls_branch_max)
        out = self.upsample(self.upsample(out))
        return out

class Unet3plus(nn.Module):
    def __init__(self, num_classes, encoder="convnext_tiny"):
        super().__init__()
        enc =  timm.create_model(encoder, pretrained = True)
        self.econv0 = enc.stem          #96   ,256X256
        self.econv1 = enc.stages[0]     #96
        self.econv2 = enc.stages[1]     #192
        self.econv3 = enc.stages[2]     #384
        self.econv4 = enc.stages[3]     #768
    
        self.dconv4 =  double_conv(768, 384)  #,
        self.single4 = single_conv(1056,384)
        
        self.dconv3 =  double_conv(384, 192)
        self.single3 = single_conv(480,192)
        
        self.dconv2 = double_conv(192, 96)
        self.dconv1 = double_conv(192, 64)
        #self.dconv0 = double_conv(96+64, 32)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.drop = nn.Dropout2d(0.1)
        self.segment_head =  nn.Conv2d(64, num_classes, 3, padding = 'same')
        
    def forward(self, x):
             
        conv0 = self.econv0(x) # 512X512 #96
        conv1 = self.econv1(conv0)# 256X256 #96
        conv2 = self.econv2(conv1) #192
        conv3 = self.econv3(conv2)# 64X64 #384
        conv4 = self.econv4(conv3)# 32X32 #768
        
        dec4 = self.dconv4(conv4)
        dec4 = torch.cat([self.upsample(dec4), conv3, self.maxpool(conv2), self.maxpool(self.maxpool(conv1))], dim=1)# 64X64
        
        dec4 = self.single4(dec4)
        dec3 = self.dconv3(dec4)
        #dec3 = torch.cat([self.upsample(dec3), conv2[0]], dim=1)# 128X128
        dec3 = torch.cat([self.upsample(dec3), conv2,self.maxpool(conv1)], dim=1)# 128X128
        
        dec3 = self.single3(dec3)
        dec2 = self.dconv2(dec3)        
        dec2 = torch.cat([self.upsample(dec2), conv1], dim=1)# 256X256
        dec2 = self.upsample(self.upsample(dec2))
        dec1 = self.dconv1(dec2)
        dec1 = self.drop(dec1)
        out = self.segment_head(dec1)
        return out
    
    
class Unet3plus_deepsupervision(nn.Module):
    def __init__(self, num_classes, encoder="convnext_tiny"):
        super().__init__()
        enc =  timm.create_model(encoder, pretrained = True)
        self.econv0 = enc.stem          #96   ,256X256
        self.econv1 = enc.stages[0]     #96
        self.econv2 = enc.stages[1]     #192
        self.econv3 = enc.stages[2]     #384
        self.econv4 = enc.stages[3]     #768
    
        self.dconv4 =  double_conv(768, 384)  #,
        self.single4 = single_conv(1056,384)
        
        self.dconv3 =  double_conv(384, 192)
        self.single3 = single_conv(480,192)
        
        self.dconv2 = double_conv(192, 96)
        self.dconv1 = double_conv(192, 64)
        #self.dconv0 = double_conv(96+64, 32)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.drop = nn.Dropout2d(0.1)
        self.segment_head1 =  nn.Conv2d(64, num_classes, 3, padding = 'same')
        self.segment_head2 =  nn.Conv2d(96, num_classes, 3, padding = 'same')
        self.segment_head3 =  nn.Conv2d(192, num_classes, 3, padding = 'same')
        
    def forward(self, x):
             
        conv0 = self.econv0(x) # 512X512 #96
        conv1 = self.econv1(conv0)# 256X256 #96
        conv2 = self.econv2(conv1) #192
        conv3 = self.econv3(conv2)# 64X64 #384
        conv4 = self.econv4(conv3)# 32X32 #768
        
        dec4 = self.dconv4(conv4)
        dec4 = torch.cat([self.upsample(dec4), conv3, self.maxpool(conv2), self.maxpool(self.maxpool(conv1))], dim=1)# 64X64
        
        dec4 = self.single4(dec4)
        dec3 = self.dconv3(dec4)
    
        out3 = self.upsample(dec3)
        out3 = self.drop(out3)
        out3 = self.segment_head3(out3)
        out3 = self.upsample(out3)
    
        dec3 = torch.cat([self.upsample(dec3), conv2,self.maxpool(conv1)], dim=1)# 128X128
        
        dec3 = self.single3(dec3)
        dec2 = self.dconv2(dec3)
        
        out2 = self.upsample(dec2)
        out2 = self.drop(out2)
        out2 = self.segment_head2(out2)
        out2 = self.upsample(out2)
        
        
        dec2 = torch.cat([self.upsample(dec2), conv1], dim=1)# 256X256
        dec2 = self.upsample(self.upsample(dec2))
        dec1 = self.dconv1(dec2)
        dec1 = self.drop(dec1)
        out1 = self.segment_head1(dec1)
        return out1,out2,out3
