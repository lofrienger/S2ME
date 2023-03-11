import json
import os

import torch
from torch.utils.data import Dataset

from .preprocess_polyp import *

# statistics from SUN-SEG dataset: https://github.com/GewelsJI/VPS/blob/main/lib/dataloader/statistics.pth
MEAN = np.array([0.4732661 , 0.44874457, 0.3948762 ], dtype=np.float32)
STD = np.array([0.22674961, 0.22012031, 0.2238305 ], dtype=np.float32)

trsf_train_image_224 = Compose_imglabel([
    Resize_image(224, 224),
    Random_crop_Resize_image(7),
    Random_horizontal_flip_image(0.5),
    Random_vertical_flip_image(0.5),
    toTensor_image(),
    Normalize_image(MEAN, STD)
])

trsf_valid_image_224 = Compose_imglabel([
    Resize_image(224, 224),
    toTensor_image(),
    Normalize_image(MEAN, STD)
])


class PolypDataset(Dataset):
    def __init__(self, ds_root, csv_root, split='train', label='GT', transform=None):
        super(PolypDataset, self).__init__()
        self.ds_root = ds_root
        self.label = label
        self.split = split
        self.transform = transform

        self.image_paths = self.get_frames_from_csv(
            os.path.join(csv_root, split+'_frames.json'))
        self.image_len = len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.ds_root, self.image_paths[idx])
        label_path = image_path.replace(
            ".jpg", ".png").replace('Frame', self.label)
        image_path = os.path.join(self.ds_root, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform != None:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return self.image_len

    def get_frames_from_csv(self, csv_path):
        frame_paths = []
        frame_num = 0
        with open(csv_path) as csv:
            data_dict = json.load(csv)
        case_list = list(data_dict.keys())
        for case in case_list:
            frame_paths.extend(data_dict[case][0])
            frame_num += data_dict[case][1]
        assert len(frame_paths) == frame_num, 'len(frame_paths) != frame_num'

        return frame_paths

class SUNDataset(Dataset):
    def __init__(self, ds_root, csv_root, split='train', label='GT', transform=None):
        super(SUNDataset, self).__init__()
        self.ds_root = ds_root
        self.label = label
        self.split = split
        self.transform = transform

        self.image_paths = self.get_frames_from_csv(
            os.path.join(csv_root, split+'_frames.json'))
        self.image_len = len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.ds_root, self.image_paths[idx])
        label_path = image_path.replace(
            ".jpg", ".png").replace('Frame', self.label)
        image_path = os.path.join(self.ds_root, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform != None:
            image, label = self.transform(image, label)
        if self.split == 'test':
            return image, label, image_path
        else:
            return image, label

    def __len__(self):
        return self.image_len

    def get_frames_from_csv(self, csv_path):
        frame_paths = []
        frame_num = 0
        with open(csv_path) as csv:
            data_dict = json.load(csv)
        case_list = list(data_dict.keys())
        for case in case_list:
            frame_paths.extend(data_dict[case][0])
            frame_num += data_dict[case][1]
        assert len(frame_paths) == frame_num, 'len(frame_paths) != frame_num'

        return frame_paths

class KvasirDataset(Dataset): #/mnt/data-hdd/wa/dataset/Polyp/Kvasir-SEG
    def __init__(self, ds_root, transform=None, save='False'):
        super(KvasirDataset, self).__init__()
        self.ds_root = ds_root
        self.transform = transform
        self.save = save

        self.image_paths = os.listdir(os.path.join(self.ds_root, 'images'))
        self.image_len = len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.ds_root, 'images', self.image_paths[idx])
        label_path = image_path.replace('images', 'masks')
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform != None:
            image, label = self.transform(image, label)
        if self.save == 'True':
            return image, label, image_path
        else:
            return image, label

    def __len__(self):
        return self.image_len

class CVCDataset(Dataset): #/mnt/data-hdd/wa/dataset/Polyp/CVC-ClinicDB/PNG
    def __init__(self, ds_root, transform=None, save='False'):
        super(CVCDataset, self).__init__()
        self.ds_root = ds_root
        self.transform = transform
        self.save = save

        self.image_paths = os.listdir(os.path.join(self.ds_root, 'Original'))
        self.image_len = len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.ds_root, 'Original', self.image_paths[idx])
        label_path = image_path.replace('Original', 'Ground Truth')
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform != None:
            image, label = self.transform(image, label)

        if self.save == 'True':
            return image, label, image_path
        else:
            return image, label

    def __len__(self):
        return self.image_len
    
class PolypGenDataset(Dataset): #/mnt/data-hdd/wa/dataset/Polyp/PolypGen
    def __init__(self, ds_root, transform=None, save='False'):
        super(PolypGenDataset, self).__init__()
        self.ds_root = ds_root
        self.transform = transform
        self.image_paths = []
        self.save = save

        data_c = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        for dc in data_c:
            images = os.listdir(os.path.join(self.ds_root, 'data_'+dc, 'images_'+dc))
            self.image_paths.extend(self.ds_root + '/data_'+dc + '/images_'+ dc + '/' + image for image in images)

        self.image_len = len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = image_path.replace('images_C', 'masks_C')[:-4]+'_mask.jpg'
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform != None:
            image, label = self.transform(image, label)

        if self.save == 'True':
            return image, label, image_path
        else:
            return image, label

    def __len__(self):
        return self.image_len