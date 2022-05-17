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
    device = torch.device('cpu') 
    l1_loss = torch.nn.L1Loss().to(device)

    real1_img = read_image(str(Path('/mnt/c/Users/takagi yugo/Nextcloud/AIR-JAIST/10_M1-研究室/01_研究会/20220419/_2022/CUT/real') / '2021-12-27_11_57_53.076322image1.png'))
    real2_img = read_image(str(Path('/mnt/c/Users/takagi yugo/Nextcloud/AIR-JAIST/10_M1-研究室/01_研究会/20220419/_2022/CUT/real') / '2021-12-27_11_58_08.074410image1.png'))
    fake1_img = read_image(str(Path('/mnt/c/Users/takagi yugo/Nextcloud/AIR-JAIST/10_M1-研究室/01_研究会/20220419/_2022/CUT/CUT') / '2021-12-27_11_57_53.076322image1.png'))
    fake2_img = read_image(str(Path('/mnt/c/Users/takagi yugo/Nextcloud/AIR-JAIST/10_M1-研究室/01_研究会/20220419/_2022/CUT/CUT') / '2021-12-27_11_58_08.074410image1.png'))

    print('real1_img.shape:', real1_img.shape)
    print('real2_img.shape:', real2_img.shape)
    print('fake1_img.shape:', fake1_img.shape)
    print('fake2_img.shape:', fake2_img.shape)
    
    # channelsを3にする．
#    real1_img, img_ = torch.split(real1_img, 3)
#    real2_img, img_ = torch.split(real2_img, 3)
#    fake1_img, img_ = torch.split(fake1_img, 3)
#    fake2_img, img_ = torch.split(fake2_img, 3)
    
    # 画像の上1/3を切り抜き．
    crop_height = int(real1_img.shape[1] / 3)
    
    real1_img_top = top_crop(real1_img, crop_height)
    real2_img_top = top_crop(real2_img, crop_height)
    fake1_img_top = top_crop(fake1_img, crop_height)
    fake2_img_top = top_crop(fake2_img, crop_height)
    
    real1_img_top = (real1_img_top * 1.0)
    real2_img_top = (real2_img_top * 1.0)
    fake1_img_top = (fake1_img_top * 1.0)
    fake2_img_top = (fake2_img_top * 1.0)

    # 画像をグレーにする．
    gray_real1_img_top = T.Grayscale(num_output_channels=3)(real1_img_top)
    gray_real2_img_top = T.Grayscale(num_output_channels=3)(real2_img_top)
    gray_fake1_img_top = T.Grayscale(num_output_channels=3)(fake1_img_top)
    gray_fake2_img_top = T.Grayscale(num_output_channels=3)(fake2_img_top)

    gray_real1_img_top = (gray_real1_img_top * 1.0)
    gray_real2_img_top = (gray_real2_img_top * 1.0)
    gray_fake1_img_top = (gray_fake1_img_top * 1.0)
    gray_fake2_img_top = (gray_fake2_img_top * 1.0)

    # L1損失を計算．
    print('l1_loss(real1_img_top, real1_img_top)):', l1_loss(real1_img_top, real1_img_top))
    print('l1_loss(real1_img_top, fake1_img_top)):', l1_loss(real1_img_top, fake1_img_top))
    print('l1_loss(real2_img_top, fake2_img_top)):', l1_loss(real2_img_top, fake2_img_top))
    
    print('l1_loss(gray_real1_img_top, gray_fake1_img_top)):', l1_loss(gray_real1_img_top, gray_fake1_img_top))
    print('l1_loss(gray_real2_img_top, gray_fake2_img_top)):', l1_loss(gray_real2_img_top, gray_fake2_img_top))
    
    print('l1_loss(real1_img_top, real2_img_top)):', l1_loss(real1_img_top, real2_img_top))
    print('l1_loss(gray_real1_img_top, gray_fake2_img_top)):', l1_loss(gray_real1_img_top, gray_fake2_img_top))


    #show(img_ch3)
    #show(gray_img)
    #plt.show()

def _l1_loss(img1, img2):
    #loss = torch.abs(img1 - img2).mean()
    loss = (torch.abs(img1 - img2) * 1.0).mean()
    #print(abs(img1 - img2))
    #loss = torch.mean(torch.abs(img1 - img2), 3)
    return loss

if __name__ == '__main__':
    main()
