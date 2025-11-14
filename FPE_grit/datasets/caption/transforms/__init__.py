from torchvision import transforms
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, CenterCrop
from .randaug import RandAugment
from .utils import MinMaxResize, MaxWHResize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESIZE = {'normal': Resize, 'minmax': MinMaxResize, 'maxwh': MaxWHResize}  #通常会将输入图像缩放，使其宽度和高度都不超过指定的最大值（max width 和 max height），但保持原始的宽高比。


def denormalize():
    dmean = [-m / s for m, s in zip(MEAN, STD)]
    dstd = [1 / s for s in STD]
    return Compose([Normalize(mean=dmean, std=dstd), transforms.ToPILImage()])


def normalize():
    return transforms.Normalize(mean=MEAN, std=STD)


def get_transform(cfg):
    resize = RESIZE[cfg.resize_name](cfg.size)
    if cfg.randaug:
        return {
            # 'train': Compose([resize, RandAugment(), ToTensor(), normalize()]),
            'train': Compose([resize, ToTensor(), normalize()]),
            # resize: 调整图像大小。
            # RandAugment(): 应用随机增强，这是一种数据增强技术，通过随机选择和应用一系列增强操作来增加图像的多样性。
            # ToTensor(): 将图像转换为 PyTorch 张量。
            # normalize: 归一化图像，通常是将图像像素值缩放到 [0, 1] 或者进行标准化。
            'valid': Compose([resize, ToTensor(), normalize()]),   # 无随机增强
        }
    else:
        return {
            'train': Compose([resize, ToTensor(), normalize()]),
            'valid': Compose([resize, ToTensor(), normalize()]),
        }


