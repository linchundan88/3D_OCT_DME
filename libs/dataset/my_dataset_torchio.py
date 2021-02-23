import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms
import numpy as np
import torchio as tio
from torch.utils.data import DataLoader
import random


class Dataset_CSV(Dataset):
    def __init__(self, csv_file, channel_first=True, image_shape=None, test_mode=False,
                 resample_ratio=(1, 1, 1),  crop_pad_pixel=15, crop_pad_ratio=(3, 9),
                 imgaug_iaa=None,
                 ):
        assert os.path.exists(csv_file), f'csv file {csv_file} does not exists'
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        assert len(self.df) > 0, 'csv file is empty!'
        self.image_shape = image_shape
        self.imgaug_iaa = imgaug_iaa
        self.channel_first = channel_first
        self.test_mode = test_mode

        self.resample_ratio = resample_ratio
        self.crop_pad_pixel = crop_pad_pixel
        self.crop_pad_ratio = crop_pad_ratio

    def __getitem__(self, index):
        file_npy = self.df.iloc[index][0]
        assert os.path.exists(file_npy), f'npy file {file_npy} does not exists'
        array_npy = np.load(file_npy)  # shape (D,H,W)
        if array_npy.ndim > 3:
            array_npy = np.squeeze(array_npy)

        int_r = random.randint(0, 20)
        if int_r % 2 == 0:
            array_4d = np.expand_dims(array_npy[0::2, :, :], axis=0)
        else:
            array_4d = np.expand_dims(array_npy[1::2, :, :], axis=0)

        subject1 = tio.Subject(
            oct=tio.ScalarImage(tensor=array_4d),
        )
        subjects_list = [subject1]

        crop_a = random.randint(self.crop_pad_ratio[0], self.crop_pad_ratio[1])
        crop_b = self.crop_pad_pixel - crop_a
        pad_a = random.randint(self.crop_pad_ratio[0], self.crop_pad_ratio[1])
        pad_b = self.crop_pad_pixel - pad_a
        transform_1 = tio.Compose([
            # tio.OneOf({
            #     tio.RandomAffine(): 0.8,
            #     tio.RandomElasticDeformation(): 0.2,
            # }, p=0.75,),
            # tio.RandomGamma(log_gamma=(-0.3, 0.3)),

            tio.RandomFlip(axes=2, flip_probability=0.5),
            # tio.RandomAffine(
            #     scales=(0, 0, 0.9, 1.1, 0, 0), degrees=(0, 0, -5, 5, 0, 0),
            #     image_interpolation='nearest'),

            tio.RandomNoise(std=(0, 0.1)),
            tio.Crop(cropping=(0, 0, crop_a, crop_b, 0, 0)),  # (d,h,w) crop height
            tio.Pad(padding=(0, 0, pad_a, pad_b, 0, 0)),
            tio.Resample(self.resample_ratio),
            # tio.RescaleIntensity((0, 255))
        ])

        if random.randint(1, 10) == 5:
            transform = tio.Compose([tio.Resample(self.resample_ratio)])
        else:
            transform = transform_1

        subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)

        inputs = subjects_dataset[0]['oct'][tio.DATA]
        array1 = np.squeeze(inputs.cpu().numpy())  #array1.shape: (D,H,W)
        array1 = array1.astype(np.uint8)

        if self.imgaug_iaa is not None:
            self.imgaug_iaa.deterministic = True
        else:
            if (array1.shape[1:3]) == (self.image_shape[0:2]):  # (H,W)
                array_4d_out = np.expand_dims(array_npy, axis=-1)  #(D,H,W,C)

        if 'array_4d_out' not in locals().keys():
            list_images = []
            for i in range(array1.shape[0]):  # D,H,W
                img = array1[i, :, :]
                if (img.shape[0:2]) != (self.image_shape[0:2]):  # (H,W)
                    img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))  # resize(width,height)

                # cvtColor do not support float64
                img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2BGR)
                # other wise , MultiplyBrightness error
                img = img.astype(np.uint8)
                if self.imgaug_iaa is not None:
                    img = self.imgaug_iaa(image=img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                list_images.append(img)

            array_4d_out = np.array(list_images)  # (D,H,W)
            array_4d_out = np.expand_dims(array_4d_out, axis=-1) #(D,H,W,C)

        if self.imgaug_iaa is not None:
            self.imgaug_iaa.deterministic = False

        if self.channel_first:
            array_4d_out = np.transpose(array_4d_out, (3, 0, 1, 2))

        array_4d_out = array_4d_out.astype(np.float32)
        array_4d_out = array_4d_out / 255.

        tensor = torch.from_numpy(array_4d_out)

        if self.test_mode:
            return tensor
        else:
            label = int(self.df.iloc[index][1])
            return tensor, label

    def __len__(self):
        return len(self.df)

