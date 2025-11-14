import torch
import torch.nn as nn
import torch.fft as fft
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple
from PIL import Image

import cv2
from torchvision.transforms import functional as F
from torchvision.transforms import ToTensor, Compose, Normalize, Resize , CenterCrop

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def denormalize():
    dmean = [-m / s for m, s in zip(MEAN, STD)]
    dstd = [1 / s for s in STD]
    # return Compose([Normalize(mean=dmean, std=dstd), T.ToPILImage()])  
    return Compose([Normalize(mean=dmean, std=dstd)])

def normalize():
    return T.Normalize(mean=MEAN, std=STD)




class MaxWHResize:

    def __init__(self, size):
        self.size = size
        self.max_h = size[0]
        self.max_w = size[1]

    def __call__(self, x):
        w, h = x.size
        scale = min(self.max_w / w, self.max_h / h)
        neww = int(w * scale)
        newh = int(h * scale)
        return x.resize((neww, newh), resample=Image.BICUBIC)



# ====================================
def apply_fft(images, device):

    fft_shifted = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
    return fft_shifted.to(device)

def create_gaussian_filter(h, w, D0, high_pass=False):

    center_h, center_w = h//2, w//2
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist_sq = (x - center_w)**2 + (y - center_h)**2
    H = torch.exp(-dist_sq / (2 * D0**2))
    return 1 - H if high_pass else H

def frequency_filter(fft_data, D0=30, high_pass=False):

    B, C, H, W = fft_data.shape
    device = fft_data.device
    
    H_filter = create_gaussian_filter(H, W, D0, high_pass).to(device)
    H_filter = H_filter.view(1, 1, H, W).expand(B, C, -1, -1)

    fft_result = fft_data * H_filter
    
    return fft_result

def inverse_fft(filtered_fft):

    fft_ishifted = torch.fft.ifftshift(filtered_fft, dim=(-2, -1))
    return torch.fft.ifft2(fft_ishifted, dim=(-2, -1)).real


class Get_frequency_map(nn.Module):
    def __init__(self, cut_off_frequency, filter_model: int = 2 ):  #   [0,1,2] means [low, high, both]
        if int(filter_model) = 0:
            self.cut_frequency_low = cut_off_frequency[0]
            self.cut_frequency_high = cut_off_frequency[0]
        elif int(filter_model) = 1:
            self.cut_frequency_low = cut_off_frequency[1]
            self.cut_frequency_high = cut_off_frequency[1]
        else:
            self.cut_frequency_low = cut_off_frequency[0]
            self.cut_frequency_high = cut_off_frequency[1]

        self.read_image_from_this_main=False

    
    def get_img_fft(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        device = img_tensor.device
    
        if not self.read_image_from_this_main:
            
            denormalizer = denormalize()
            img_tensor = denormalizer(img_tensor)

        batch_image_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
        
        batch_fft = apply_fft(batch_image_tensor, device)
        low_fft = frequency_filter(batch_fft, D0=self.cut_frequency_low, high_pass=False)
        high_fft = frequency_filter(batch_fft, D0=self.cut_frequency_high, high_pass=True)
        
        low_imgs = inverse_fft(low_fft)
        high_imgs = inverse_fft(high_fft)
        
        # self.visualize(
        #     batch_image_tensor[0], 
        #     low_imgs[0], 
        #     high_imgs[0]
        # )
        
        # scale [0~255]，rgb
        if int(filter_model) = 0:
            return  low_imgs[0] 
        elif int(filter_model) = 1:
            return  high_imgs[0] 
        else:
            return torch.cat([low_imgs[0], high_imgs[0]], dim=0)
    
    def visualize(self, 
                 orig_img: torch.Tensor, 
                 low_img: torch.Tensor, 
                 high_img: torch.Tensor):

        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        

        orig_img_np = orig_img.permute(1, 2, 0).cpu().numpy()
        low_img_np = low_img.permute(1, 2, 0).cpu().numpy()
        high_img_np = high_img.permute(1, 2, 0).cpu().numpy()
        
        
        def normalize(img):
            return (img - img.min()) / (img.max() - img.min())
        

        axes[0].imshow(normalize(orig_img_np)) 
        axes[0].set_title("Original Image")
        axes[1].imshow(normalize(low_img_np))
        axes[1].set_title("Low-Pass Image")
        axes[2].imshow(normalize(high_img_np))
        axes[2].set_title("High-Pass Image")
        

        plt.tight_layout()
        plt.savefig('frequency_analysis.png')
        plt.show()  
        plt.close(fig)


# ====================================
def load_image(image_path: str, target_size: Tuple[int, int] = (384, 640)) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image is not in: {image_path}")
    
    img = Image.open(image_path).convert('RGB')

    # transform = T.Compose([
    #     T.Resize(target_size),
    #     T.ToTensor(),
    # ])

    # remove randaug
    resize = MaxWHResize(target_size)
    transform = T.Compose([
        resize,
        T.ToTensor(),
        normalize()
    ])


    return transform(img).float()



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    freq_map = Get_frequency_map(cut_off_frequency=30)
    
    image_path = "/gemini/code/000000039769.jpg"  
    
    try:
        img_tensor = load_image(image_path).to(device)
        print(f"load success: {image_path}")
    except Exception as e:
        print(f"load failed: {e}")
        img_tensor = torch.rand(3, 384, 640).to(device)
    
    low_img, high_img = freq_map.get_img_fft(img_tensor)