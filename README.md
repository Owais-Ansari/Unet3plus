# UNet3plus
UNet3+

## Reference





[[2021] PVT V2: Improved Baselines with Pyramid Vision Transformer ](https://arxiv.org/abs/2106.13797.pdf)

[[2020] ConvNext: A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf)

[[2020] UNET 3+: A Full-Scale Connected UNet for Medical Image Segmentation (ICASSP 2020)](https://arxiv.org/pdf/2004.08790.pdf)







## Dependencies

- Python == 3.9
- PyTorch >= 1.10.2
- Torchvision >= 0.11.3
- CUDA >= cu113
- numpy >= 1.20.3
- Pillow >= 8.3.1
- tensorboard >= 2.9.1
- tqdm == 4.62.3
- scikit-learn == 1.2.2
- timm == 0.6.13
- opencv-python == 4.7.0.72
- albumentations == 1.3.0
- segmentation-models-pytorch ==0.3.2
- pandas == 2.0.1


**For academic communication only**


augmentation are tuned for the histpathological images
	- Equalize, HueSaturationValue, ColorJitter, Blur, RandomBrightnessContrast, ChannelShuffle
	- Transpose, RandomRotate90, Flip
	
	

**Training methodlogy**

	- Mixed Precision 
	- Gradient Accumalation
	- Label Smoothing
	- Gradient Clipping

## Output
- checkpoint.pth.tar
- bestmodel.pth.tar if best model is found 
- commandlines.txt
- log.txt
- tensorboard file to track the training curves
- train_dice_score .mat and val_dice_score.mat files which consists of path to each image and image mean dice

	
**Usage**
```shell script
from utils.models import Unet3plus, Unet3plusGlcm, Unet3plus_deepsupervision

model = Unet3plus(num_classes = num_classes, encoder = config.encoder)
```
## Run locally
**Note : Use Python 3**



### Training

```shell script
python train.py 


Train the UNet on images and target masks
Update the config.py file

optional arguments:
	"Segmentation Training"
	num_workers = 8
	epochs = 50
	train_batch = 8
	lr = 0.0005
	weight_decay = 0.0005
	checkpoint = 'exp1'
	resume = '' #path to checkpoint
	gpu = 0
	seed = 42
	clip = None # else 0.99999
	size = 512
	ignore_label = 5 # should keep same as num_classes
	accum_iter=1 #Gradient Accumalation is True if accum_iter>1
	label_sm = 0.08
	freeze_backbone = False
	encoder = 'convnext_tiny'
	num_classes  = 5
	train_image_path = '../dataset/train/images/'
	train_mask_path = '../dataset/train/masks/'
	validation_image_path = '../dataset/val/images/'
	validation_mask_path = '../dataset/val/masks/'              
```

The input images and target masks should have the same name.
The ignore label in mask should have label value = 255


## To-DO
Deep Supervision





