import os
import numpy as np
import albumentations
from torch.utils.data import Dataset, Subset
import PIL
from PIL import Image
import json
import torch
import random
import cv2
import sys
from torchvision.datasets.utils import download_url
from transformers import CLIPProcessor, CLIPModel


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def imagedataset(self, image_path, resolution=512, min_crop_f=0.5, max_crop_f=1., negative_image=None, negative_id_list=None):
        self.resolution = resolution
        self.train = True
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.resolution, interpolation=cv2.INTER_AREA)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        self.image_path = image_path
        self.negative_image = negative_image
        if self.negative_image:
            self.negative_id_list = negative_id_list
 

    def __len__(self):
        return 1


    def __getitem__(self,idx):
        example = dict()
        
        image_path = self.image_path
        image = Image.open(image_path).convert('RGB')   
        image=np.array(image)
        image = (image/127.5 - 1.0).astype(np.float32)
        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)
        self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]
        example["image"]=image

        if self.negative_image:
            image_path =  self.negative_id_list[np.random.randint(0,len(self.negative_id_list)-1)]
            image = Image.open(image_path).convert('RGB')   
            image=np.array(image)
            image = (image/127.5 - 1.0).astype(np.float32)
            min_side_len = min(image.shape[:2])
            crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
            crop_side_len = int(crop_side_len)
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
            image = self.cropper(image=image)["image"]
            image = self.image_rescaler(image=image)["image"]
            example["negativeimage"]=image
        example["caption"] = "A photo of Yann Lecun." # Initial random prompt
        return example

class CustomTrain(CustomBase):
    def __init__(self, image_path):
        super().__init__()
        
        CustomBase.imagedataset(self, image_path)


class CustomTest(CustomBase):
    def __init__(self, image_path):
        super().__init__()
        
        CustomBase.imagedataset(self, image_path)
