
import torch
import numpy as np




def get_random_crop(image,mask,crop_height=256, crop_width=256):
    'to create random crops'
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    mask = mask[y: y + crop_height, x: x + crop_width]
    return crop,mask

