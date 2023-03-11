import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor as torchtotensor

class Compose_imglabel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

class Random_crop_Resize_image(object):
    def _randomCrop(self, img, label, x, y):
        width, height = img.size
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.NEAREST)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, label):
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        res_img, res_label = self._randomCrop(img, label, x, y)
        return res_img, res_label


class Random_horizontal_flip_image(object):
    def _horizontal_flip(self, img, label):
        return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, img, label):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        if random.random() < self.prob:
            res_img, res_label = self._horizontal_flip(img, label)
            return res_img, res_label
        else:
            return img, label


class Random_vertical_flip_image(object):
    def _vertical_flip(self, img, label):
        return img.transpose(Image.FLIP_TOP_BOTTOM), label.transpose(Image.FLIP_TOP_BOTTOM)

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, img, label):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        if random.random() < self.prob:
            res_img, res_label = self._vertical_flip(img, label)
            return res_img, res_label
        else:
            return img, label

class Resize_image(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, label):
        res_img = img.resize((self.width, self.height), Image.BILINEAR)
        res_label = label.resize((self.width, self.height), Image.NEAREST)
        return res_img, res_label

class Normalize_image(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img, label):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img, label

    
class toTensor_image(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, img, label):
        res_img = self.totensor(img)
        label_np = np.array(label)
        if len(np.unique(label_np)) == 3:
            label_np[label_np==0] = 2
            label_np[label_np==127] = 0
            label_np[label_np==255] = 1
            res_label = torch.from_numpy(label_np)
        # elif len(np.unique(label_np)) == 2:
        else:
            # label_np[label_np==255] = 1
        # res_label = torch.from_numpy(label_np)
            res_label = self.totensor(label).long().squeeze(0)
        return res_img, res_label

