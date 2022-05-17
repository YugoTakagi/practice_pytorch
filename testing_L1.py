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

import glob

def main():
    device = torch.device('cpu') 
    l1_loss = torch.nn.L1Loss().to(device)
    
    dir_real = '../01_研究会/20220419/_2022/CUT/real/*'
    dir_fake = '../01_研究会/20220419/_2022/CUT/CUT/*'
    
    real_image_names = sorted( glob.glob(dir_real) )
    fake_image_names = sorted( glob.glob(dir_fake) ) 
    rin = [r.split('/')[-1] for r in real_image_names]
    fin = [r.split('/')[-1] for r in fake_image_names]
    print('-> real:\n', rin)
    print('-> fake:\n', fin, '\n')
   
    grid = []
    real_images = []
    fake_images = []
    num_image = len(real_image_names)
    for i in range(num_image):
        real_images.append( read_image(str(Path(real_image_names[i]))) )
        fake_images.append( read_image(str(Path(fake_image_names[i]))) )
        
        # channelsを3にする．
        # real1_img, img_ = torch.split(real1_img, 3)
        
        # 画像の上1/3を切り抜き．
        if i == 0:
            crop_height = int(real_images[0].shape[1] / 3)
            print('crop_height:', crop_height)
        real_images[i] = top_crop(real_images[i], crop_height) * 1.0 
        fake_images[i] = top_crop(fake_images[i], crop_height) * 1.0 
    
        # 画像をグレーにする．
        real_images[i] = T.Grayscale(num_output_channels=3)(real_images[i])
        fake_images[i] = T.Grayscale(num_output_channels=3)(fake_images[i])
        #real_images[i] = T.Grayscale(num_output_channels=1)(real_images[i])
        #fake_images[i] = T.Grayscale(num_output_channels=1)(fake_images[i])
        
        grid.append(real_images[i])
        grid.append(fake_images[i])
        #show(grid) 

        # L1損失を計算．
        print('l1_loss({}, {}) : {}'.format(rin[i], fin[i], l1_loss(real_images[i], fake_images[i]))) 
        #print('l1_loss({}, {}) : {}'.format(real_image_names[i], fake_image_names[i], l1_loss(real_images[i], fake_images[i]))) 

    #plt.show()


def _l1_loss(img1, img2):
    #loss = torch.abs(img1 - img2).mean()
    loss = (torch.abs(img1 - img2) * 1.0).mean()
    #print(abs(img1 - img2))
    #loss = torch.mean(torch.abs(img1 - img2), 3)
    return loss

if __name__ == '__main__':
    main()
