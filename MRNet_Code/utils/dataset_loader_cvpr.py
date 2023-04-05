# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

import PIL.Image
from PIL import Image
import random
from PIL import ImageEnhance
import torch
from torch.utils import data
from config import Config


# Data Augmentation
def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = [lab_i.transpose(Image.FLIP_LEFT_RIGHT) for lab_i in label]
    return img, label

def randomCrop(image, label):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), [lab_i.crop(random_region) for lab_i in label]

def randomRotation(image, label):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label = [lab_i.rotate(random_angle, mode) for lab_i in label]
    return image,label

def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


config = Config()


class MyData(data.Dataset):  # inherit
    """
    load data in a folder
    """

    def __init__(self, root, DF, transform=True, augment=False):
        super(MyData, self).__init__()
        self.root = root
        self._transform = transform
        self._augment = augment
        self.scale_size = config.SCALE_SIZE

        self.DF = pd.DataFrame(columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin',
                                        'xmax', 'ymax', 'width', 'height','discFlag','rater'])
        for spilt in DF:
            DF_all = pd.read_csv(root + '/' + 'Glaucoma_multirater_' + spilt + '.csv', encoding='gbk')

            DF_this = DF_all.loc[DF_all['rater'] == 0]      # Final Label
            DF_this = DF_this.reset_index(drop=True)
            DF_this = DF_this.drop('Unnamed: 0', 1)
            self.DF = pd.concat([self.DF, DF_this])

        self.DF.index = range(0, len(self.DF))


    def __len__(self):
        return len(self.DF)

    def __getitem__(self, index):
        img_Name = self.DF.loc[index, 'imgName']
        """ Get the images """
        fullPathName = os.path.join(self.root, img_Name)
        fullPathName = fullPathName.replace('\\', '/')  # image path

        img = PIL.Image.open(fullPathName).convert('RGB')  # read image
        img = img.resize((self.scale_size, self.scale_size))


        """ Get the six raters masks """
        masks = []
        data_path = self.root
        for n in range(1,7):     # n:1-6
            # # load rater 1-6 label recurrently

            maskName = self.DF.loc[index, 'maskName'].replace('FinalLabel','Rater'+str(n))
            fullPathName = os.path.join(data_path, maskName)
            fullPathName = fullPathName.replace('\\', '/')

            Mask = PIL.Image.open(fullPathName).convert('L')
            Mask = Mask.resize((self.scale_size, self.scale_size))
            masks.append(Mask)

        if self._augment:
            img, masks = cv_random_flip(img, masks)
            #img, masks = randomCrop(img, masks)
            img, masks = randomRotation(img, masks)

        img = img.resize((self.scale_size, self.scale_size))
        masks = [Mask.resize((self.scale_size, self.scale_size)) for Mask in masks]

        for i in range(len(masks)):
            Mask = masks[i]
            Mask = np.array(Mask)

            if Mask.max() > 1:
                Mask = Mask / 255.0

            disc = Mask.copy()
            disc[disc != 0] = 1
            cup = Mask.copy()
            cup[cup != 1] = 0
            Mask = np.stack((disc, cup))  # [2, H, W]

            # Mask = Mask.transpose((2, 0, 1))
            Mask = torch.from_numpy(Mask)
            masks[i] = Mask


        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)  # add additional channel in dim 2 (channel)
        img_ori = img


        if self._transform:
            img_ori, img, masks = self.transform(img_ori, img, masks)
            return {'image': img, 'image_ori': img_ori, 'mask': masks, 'name': img_Name.split('.')[0]}
        else:
            return {'image': img, 'image_ori': img_ori, 'mask': masks, 'name': img_Name.split('.')[0]}


    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img_o, img, lbl):
        if img.max() > 1:
            img = img.astype(np.float64) / 255.0
        img -= config.MEAN_AND_STD['mean_rgb']
        img /= config.MEAN_AND_STD['std_rgb']
        img = img.transpose(2, 0, 1)  # to verify
        img = torch.from_numpy(img)

        if img.max() > 1:
            img_o = img_o.astype(np.float64) / 255.0
        img_o = img_o.transpose(2, 0, 1)  # to verify
        img_o = torch.from_numpy(img_o)

        return img_o, img, lbl
