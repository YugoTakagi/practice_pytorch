# sphinx_gallery_thumbnail_path = "../../gallery/assets/visualization_utils_thumbnail2.png"

import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

plt.rcParams["savefig.bbox"] = 'tight'

def main():
    path = '../01_研究会/20220419/_2022/CUT/CUT/'
    img = read_image(str(Path(path) / '2021-12-27_11:57:53.076322image1.png'))
    show(img)

    plt.show()

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

if __name__ == '__main__':
    main()
