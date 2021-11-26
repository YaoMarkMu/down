import torch
from timm.data.mixup import Mixup
from timm.data.dataset import ImageDataset
from timm.data.loader import create_loader

import torch
import torch.nn as nn
import kornia.augmentation as aug
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import time
# mixup=aug.RandomMixUp(p=1)
# random_shift = nn.Sequential(nn.ReplicationPad2d(4),aug.RandomCrop((84, 84)))
pooling= nn.MaxPool1d(2, stride=2)
from einops import rearrange, reduce, repeat
from math import *

def update_pool(x):
    B=1
    L, C = x.shape
    print(L,C)
    H, W = int(sqrt(L)), int(sqrt(L))
    print(H,W)
    if int(L) == (H * W):
        x_pool = pooling(x.transpose(2, 1)).transpose(2, 1)
    else:
        print("WWWW",x.shape)
        x_pool = x.transpose(2, 1).view(B, C, H, int(W / 2)).transpose(2, 1)
        x_pool = pooling(x_pool).view(B, C, int(W / 2), int(H / 2)).transpose(2, 1)
        x_pool = x_pool.view(B, C, int(L / 4)).transpose(2, 1)
    return x_pool

def randconv(image: torch.Tensor, K: int, mix: bool, p: float) -> torch.Tensor:
    """
    Outputs the image or the random convolution applied on the image.
    Args:
        image (torch.Tensor): input image
        K (int): maximum kernel size of the random convolution
    """

    p0 = torch.rand(1).item()
    if p0 < p:
        return image
    else:
        k = torch.randint(1, K+1, (1, )).item()
        #k=K
        random_convolution = nn.Conv2d(3, 3, 2*k + 1, padding=k).to(image.device)
        torch.nn.init.uniform_(random_convolution.weight,
                              0, 1. / (3 * k * k))
        image_rc = random_convolution(image).to(image.device)

        if mix:
            alpha = 0.7
            return alpha * image + (1 - alpha) * image_rc
        else:
            return image_rc

class kaiming_aug():
    def __init__(self, b1=28,b2=28):
        super().__init__()
        self.b1=b1
        self.b2=b2
        self.length=b1*b2

    def aug_trans(self,img,up):

        img = rearrange(img, 'b c (b1 h) (b2 w) -> b c (b1 b2) h w ', b1=self.b1, b2=self.b2)
        up = rearrange(up, 'b c (b1 h) (b2 w) -> b c (b1 b2) h w ', b1=self.b1, b2=self.b2)
        # print(img.shape)
        # print(up.shape)
        mask = self.get_mask(self.length)
        # print(img[:, :, mask, :, :].shape)
        # print(up[:,:3, mask, :, :].shape)
        img[:,:, mask, :, :] = up[:,:3, mask, :, :]*np.random.normal(0,1)
        img = rearrange(img, 'b c (b1 b2) h w -> b c (b1 h)  (b2 w) ', b1=self.b1, b2=self.b2)
        return img

    def get_mask(self,p):
        mask = np.arange(0, p)
        np.random.shuffle(mask)
        mask = mask[:196]
        mask.sort()
        #print(mask)
        return mask

trans=kaiming_aug()


def pool_permute(x):
    B, C, L = x.shape
    return x.view(B, C, int(sqrt(L)), int(sqrt(L))).transpose(-1, -2).reshape(B, C, -1)


def revise( x,first):
    B, C, L = x.shape
    print(B, C, L)
    if first:
        return x
    else:

        W = int(sqrt(int(L / 2)))
        H = 2 * W
        print(":sss",x.shape)
        return x.view(B, C, H, W).transpose(3,2).reshape(B, C, -1)
while 1:
    img = torch.tensor(io.imread("ss.jpg")).float()
    up = torch.tensor(io.imread("jj.jpg")).float()
    img = torch.unsqueeze(img.permute(2, 0, 1), 0)/255.0
    up = torch.unsqueeze(up.permute(2, 0, 1), 0)
    img = img.reshape((3,-1))
    img = pooling(img)
    img = img.view(3, 42, 84)
    # img = pooling(img)
    # img = revise(torch.unsqueeze(img,0),first=False)
    # print("#",img.shape)
    # img = pooling(img)
    # print(img.shape)
    # # img = update_pool(img)
    # # img = update_pool(img)
    # # print(img.shape)
    # # print(img.shape)
    # # img=img.transpose(2,1)
    # # img = pooling(img)
    # # print(img.shape)
    # img = pooling(img).view(3,84,42).transpose(2,1).reshape((3,-1))
    # print(img.shape)
    # img = pool_permute(pooling(img))
    # print(img.shape)
    img=img.view(3,42,84)
    # # # # # img = trans.aug_trans(img,up).squeeze()
    # # # # # img=randconv(img,10, True,0.0)
    # # # # print(img.shape)
    plt.imshow(np.uint8(255.0*img.permute(1, 2, 0).detach().numpy()))
    plt.pause(0.1)
    # #




#
# cutout = aug.RandomErasing(scale=(0.05,0.05),ratio=(1.0,1.0),p=1)
# for i in range(100):
#
#     #print()
#
#     img = img.permute(2, 0, 1)
#     #print(img.shape)
#     img= rearrange(img, 'c (b1 h) (b2 w) -> c (b1 b2) h w ', b1=28,b2=28)
#     #print(img.shape)
#     # print(rearrange(img, 'c (b1 h) (b2 w) -> c (b1 b2) h w ', b1=14,b2=14).shape)
#     mask = get_mask(784)
#     img[:, mask, :,:]=img[:, mask, :,:]*0.0
#     #print(img.shape)
#
#     print(img.shape)
#     # img = img.reshape(4,14,14,36)
#     # mask = get_mask(196)
#     # img = img.reshape(4,196,36)
#     # img = img[:,mask,:].reshape(4,78,78).permute(1, 2, 0)
#     # # img = cutout(img.permute(2, 0, 1))
#     # # img = img.reshape(-1, 84, 84).permute(1, 2, 0)
#
#
#
#
