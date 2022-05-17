import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

import torchvision.transforms as T

from show_image import show
from crop_image import top_crop

def main():
    img = read_image(str(Path('/mnt/c/Users/takagi yugo/Nextcloud/AIR-JAIST/10_M1-研究室/01_研究会/20220419/_2022/images') / 'real.png'))

    print('img.shape:', img.shape)
    
    # なぜが，imgが4channels持っていたので，3chと1chに分ける．
    img_ch3, img_ch1 = torch.split(img, 3)
    print(img_ch3.shape)
    
    img_ch3 = top_crop(img_ch3, int(img_ch3.shape[1]/3))
    gray_img = T.Grayscale(num_output_channels=3)(img_ch3)
    print('gray_img.shape:', gray_img.shape)

    
    #show(img_ch3)
    show(gray_img)
    plt.show()


if __name__ == '__main__':
    main()
