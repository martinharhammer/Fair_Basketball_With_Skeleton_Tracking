import torch
from torchvision.transforms import ToTensor, Normalize
import numpy as np
from PIL import Image

class ResizeWithEqualScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR, fill_color=(0,0,0)):
        self.height = height
        self.width  = width
        self.interpolation = interpolation
        self.fill_color    = fill_color

    def __call__(self, img):
        w, h = img.size
        if self.height/self.width >= h/w:
            h = int(self.width * (h/w));  w = self.width
        else:
            w = int(self.height * (w/h)); h = self.height

        resized = img.resize((w,h), self.interpolation)
        canvas  = Image.new('RGB', (self.width, self.height), self.fill_color)
        canvas.paste(resized, ((self.width-w)//2, (self.height-h)//2))
        return canvas

class SeqTransformCompose(object):
    def __init__(self, frame_transform):
        self.frame_transform = frame_transform
        self.to_tensor       = ToTensor()    # -> [0,1]
        self.normalize       = Normalize(
            mean=[0.485,0.456,0.406],
            std =[0.229,0.224,0.225]
        )

    def __call__(self, frame_list):
        """
        Args:
            frame_list (list of PIL.Image), length = N
        Returns:
            torch.Tensor, shape = (3*N, H, W), normalized
        """
        out = []
        for frame in frame_list:
            im = self.frame_transform(frame)   # ResizeWithEqualScale
            im = self.to_tensor(im)            # (3,H,W) in [0,1]
            im = self.normalize(im)            # normalize exactly as in training
            out.append(im)
        return torch.cat(out, dim=0)           # (3*N, H, W)

