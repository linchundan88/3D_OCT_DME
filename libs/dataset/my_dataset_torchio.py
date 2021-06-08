import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import torchio as tio
import random


class Dataset_CSV_train(Dataset):
    def __init__(self, csv_file, channel_first=True, image_shape=None,
                 resample_ratio=(1, 1, 1),
                 depth_interval=2,
                 crop_pad_pixel=15, crop_pad_ratio=(3, 9),
                 imgaug_iaa=None,
                 ):
        assert os.path.exists(csv_file), f'csv file {csv_file} does not exists'
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        assert len(self.df) > 0, 'csv file is empty!'
        self.image_shape = image_shape
        self.imgaug_iaa = imgaug_iaa
        self.channel_first = channel_first

        self.resample_ratio = resample_ratio
        self.depth_interval = depth_interval
        self.crop_pad_pixel = crop_pad_pixel
        self.crop_pad_ratio = crop_pad_ratio

    def __getitem__(self, index):
        file_npy = self.df.iloc[index][0]
        assert os.path.exists(file_npy), f'npy file {file_npy} does not exists'
        array_npy = np.load(file_npy)  # shape (D,H,W)
        if array_npy.ndim > 3:
            array_npy = np.squeeze(array_npy)
        array_npy = np.expand_dims(array_npy, axis=0)  #(C,D,H,W)

        #if depth_interval==2  (128,128,128)->(64,128,128)
        depth_start_random = random.randint(0, 20) % self.depth_interval
        array_npy = array_npy[:, depth_start_random::self.depth_interval, :, :]

        subject1 = tio.Subject(
            oct=tio.ScalarImage(tensor=array_npy),
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
        array_3d = np.squeeze(inputs.cpu().numpy())  #shape: (D,H,W)
        array_3d = array_3d.astype(np.uint8)

        if self.imgaug_iaa is not None:
            self.imgaug_iaa.deterministic = True
        else:
            if (self.image_shape is None) or\
                    (array_3d.shape[1:3]) == (self.image_shape[0:2]):  # (H,W)
                array_4d = np.expand_dims(array_3d, axis=-1)  #(D,H,W,C)

        if 'array_4d' not in locals().keys():
            list_images = []
            for i in range(array_3d.shape[0]):
                img = array_3d[i, :, :]  #(H,W)
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

            array_4d = np.array(list_images)  # (D,H,W)
            array_4d = np.expand_dims(array_4d, axis=-1) #(D,H,W,C)

        if self.imgaug_iaa is not None:
            self.imgaug_iaa.deterministic = False

        if self.channel_first:
            array_4d = np.transpose(array_4d, (3, 0, 1, 2)) #(D,H,W,C)->(C,D,H,W)

        array_4d = array_4d.astype(np.float32)
        array_4d = array_4d / 255.
        if array_4d.shape != (1, 64, 64, 64):
            print(file_npy)

        # https://pytorch.org/docs/stable/data.html
        # It is generally not recommended to return CUDA tensors in multi-process loading because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing (see CUDA in multiprocessing).
        # tensor_x = torch.from_numpy(array_4d)

        label = int(self.df.iloc[index][1])

        return array_4d, label

    def __len__(self):
        return len(self.df)


class Dataset_CSV_test(Dataset):
    def __init__(self, csv_file, channel_first=True, image_shape=None,
                 depth_start=0, depth_interval=1,
                 test_mode=False):
        assert os.path.exists(csv_file), f'csv file {csv_file} does not exists'
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        assert len(self.df) > 0, 'csv file is empty!'
        self.image_shape = image_shape
        self.depth_start = depth_start
        self.depth_interval = depth_interval

        self.channel_first = channel_first
        self.test_mode = test_mode

    def __getitem__(self, index):
        file_npy = self.df.iloc[index][0]
        assert os.path.exists(file_npy), f'npy file {file_npy} does not exists'
        array_3d = np.load(file_npy)  # shape (D,H,W)
        if array_3d.ndim > 3:
            array_3d = np.squeeze(array_3d)
        if not(self.depth_start != 0 and self.depth_interval != 1):
            array_3d = array_3d[self.depth_start::self.depth_interval, :, :]

        if (self.image_shape is None) or \
                (array_3d.shape[1:3]) == (self.image_shape[0:2]):  # (H,W)
                array_4d = np.expand_dims(array_3d, axis=-1)  #(D,H,W,C)
        else:
            list_images = []
            for i in range(array_3d.shape[0]):
                img = array_3d[i, :, :]  #(H,W)
                if (img.shape[0:2]) != (self.image_shape[0:2]):  # (H,W)
                    img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))  # resize(width,height)

                # cvtColor do not support float64
                img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2BGR)
                # other wise , MultiplyBrightness error
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                list_images.append(img)

            array_4d = np.array(list_images)  # (D,H,W)
            array_4d = np.expand_dims(array_4d, axis=-1) #(D,H,W,C)

        if self.channel_first:
            array_4d = np.transpose(array_4d, (3, 0, 1, 2)) #(D,H,W,C)->(C,D,H,W)

        array_4d = array_4d.astype(np.float32)
        array_4d = array_4d / 255.

        # https://pytorch.org/docs/stable/data.html
        # It is generally not recommended to return CUDA tensors in multi-process loading because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing (see CUDA in multiprocessing).
        # tensor_x = torch.from_numpy(array_4d)

        if self.test_mode:
            return array_4d
        else:
            label = int(self.df.iloc[index][1])
            return array_4d, label

    def __len__(self):
        return len(self.df)


def get_tensor(file_npy, channel_first=True, image_shape=None,
                 depth_start=0, depth_interval=1):
    assert os.path.exists(file_npy), f'npy file {file_npy} does not exists'
    array_3d = np.load(file_npy)  # shape (D,H,W)
    if array_3d.ndim > 3:
        array_3d = np.squeeze(array_3d)
    if not (depth_start != 0 and depth_interval != 1):
        array_3d = array_3d[depth_start::depth_interval, :, :]

    if (image_shape is None) or (array_3d.shape[1:3]) == (image_shape[0:2]):  # (H,W)
        array_4d = np.expand_dims(array_3d, axis=-1)  # (D,H,W,C)

    if 'array_4d' not in locals().keys():
        list_images = []
        for i in range(array_3d.shape[0]):
            img = array_3d[i, :, :]  # (H,W)
            if (img.shape[0:2]) != (image_shape[0:2]):  # (H,W)
                img = cv2.resize(img, (image_shape[1], image_shape[0]))  # resize(width,height)

            # cvtColor do not support float64
            img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2BGR)
            # other wise , MultiplyBrightness error
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            list_images.append(img)

        array_4d = np.array(list_images)  # (D,H,W)
        array_4d = np.expand_dims(array_4d, axis=-1)  # (D,H,W,C)

    if channel_first:
        array_4d = np.transpose(array_4d, (3, 0, 1, 2))  # (D,H,W,C)->(C,D,H,W)

    array_5d = np.expand_dims(array_4d, axis=0)  #(C,D,H,W) (N,C,D,H,W)
    array_5d = array_5d.astype(np.float32)
    array_5d = array_5d / 255.

    tensor_x = torch.from_numpy(array_5d)

    return tensor_x


