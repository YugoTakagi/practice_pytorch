import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

from show_image import show

plt.rcParams["savefig.bbox"] = 'tight'

def main():
    img = read_image(str(Path('/mnt/c/Users/takagi yugo/Nextcloud/AIR-JAIST/10_M1-研究室/01_研究会/20220419/_2022/images') / 'real.png'))
    #show(img)
    #plt.show()

    print('type(img)', type(img))
    # image.shape : [channels, height, width]
    height = img.shape[1]
    width = img.shape[2]
    crop_height = int(height/3)

    # torchvision.transforms.functional.crop(img: torch.Tensor, top: int, left: int, height: int, width: int) → torch.Tensor
    #croped_img = torchvision.transforms.functional.crop(img, top=0, left=0, height=crop_height, width=width)
    
    croped_img = top_crop(img, crop_height)

    show(croped_img)
    # 2つの画像を表示する．(同じサイズでないとダメ．)
    #grid = make_grid([img, croped_img])
    #show(grid)

    plt.show()


def top_crop(img: torch.Tensor, crop_height: int) -> torch.Tensor:
    # torchvision.transforms.functional.crop(img: torch.Tensor, top: int, left: int, height: int, width: int) → torch.Tensor
    width = img.shape[2]
    croped_img = torchvision.transforms.functional.crop(img, top=0, left=0, height=crop_height, width=width)
    return croped_img


if __name__ == '__main__':
    main()
