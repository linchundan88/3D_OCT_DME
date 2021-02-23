import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms
import numpy as np

class Dataset_CSV_3d(Dataset):
    def __init__(self, csv_file, channel_first=True, image_shape=None, test_mode=False,
                 imgaug_iaa=None):
        assert os.path.exists(csv_file), f'csv file {csv_file} does not exists'
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        assert len(self.df) > 0, 'csv file is empty!'
        self.image_shape = image_shape
        self.imgaug_iaa = imgaug_iaa
        self.channel_first = channel_first
        self.test_mode = test_mode

    def __getitem__(self, index):
        file_npy = self.df.iloc[index][0]
        assert os.path.exists(file_npy), f'npy file {file_npy} does not exists'
        array_3d = np.load(file_npy)  # shape (D,H,W)
        if array_3d.ndim > 3:
            array_3d = np.squeeze(array_3d)

        if self.imgaug_iaa is not None:
            self.imgaug_iaa.deterministic = True
        else:
            if (array_3d.shape[1:3]) == (self.image_shape[0:2]):  # (H,W)
                array_4d = np.expand_dims(array_3d, axis=-1)

        if 'array_4d' not in locals().keys():
            list_images = []
            for i in range(array_3d.shape[0]):  # D,H,W
                img = array_3d[i, :, :]
                img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))  # resize(width,height)

                # cvtColor do not support float64
                img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2BGR)
                # other wise , MultiplyBrightness error
                img = img.astype(np.uint8)
                if self.imgaug_iaa is not None:
                    img = self.imgaug_iaa(image=img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                list_images.append(img)

            array_4d = np.array(list_images)  #(D,H,W)
            array_4d = np.expand_dims(array_4d, axis=-1) #(D,H,W,C)

        if self.imgaug_iaa is not None:
            self.imgaug_iaa.deterministic = False

        if self.channel_first:
            array_4d = np.transpose(array_4d, (3, 0, 1, 2))

        array_4d = array_4d.astype(np.float32)
        array_4d = array_4d / 255.

        tensor = torch.from_numpy(array_4d)

        if self.test_mode:
            return tensor
        else:
            label = int(self.df.iloc[index][1])
            return tensor, label

    def __len__(self):
        return len(self.df)



def npy_to_tensor(file_npy, image_shape=(64, 64), channel_first=True):
    assert os.path.exists(file_npy), f'npy file {file_npy} does not exists'
    array_3d = np.load(file_npy)  # shape (D,H,W)

    if (array_3d.shape[1:3]) != (image_shape[0:2]):  # (H,W)
        array_4d = np.expand_dims(array_3d, axis=-1)

    if 'array_4d' not in locals().keys():
        list_images = []
        for i in range(array_3d.shape[0]):  # D,H,W
            img = array_3d[i, :, :]
            if (img.shape[0:2]) != (image_shape[0:2]):  # (H,W)
                img = cv2.resize(img, (image_shape[1], image_shape[0]))  # resize(width,height)

            # cvtColor do not support float64
            img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2BGR)
            # other wise , MultiplyBrightness error
            img = img.astype(np.uint8)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite(f'/tmp2/a/{i}.jpg', img)  #code_test code
            img = np.expand_dims(img, axis=-1)  # (H,W,C)
            list_images.append(img)
            array_4d = np.array(list_images)  # (D,H,W,C)


    if channel_first:
        array_4d = np.transpose(array_4d, (3, 0, 1, 2))  #(C,D,H,W)

    array_4d = array_4d.astype(np.float32)
    array_4d = array_4d / 255.

    #(B,C,D,H,W)
    tensor = torch.from_numpy(np.expand_dims(array_4d, axis=0))

    return tensor


'''

# image2 = A.RandomRotate90(p=1)(image=image)['image']  #albumentations

from imgaug import augmenters as iaa
imgaug_iaa = iaa.Sequential([
    # iaa.CropAndPad(percent=(-0.04, 0.04)),
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Flipud(0.2),  # horizontally flip 50% of the images

    # iaa.GaussianBlur(sigma=(0.0, 0.3)),
    # iaa.MultiplyBrightness(mul=(0.8, 1.2)),
    # iaa.contrast.LinearContrast((0.8, 1.2)),
    # iaa.Sometimes(0.9, iaa.Add((-8, 8))),
    # iaa.Sometimes(0.9, iaa.Affine(
    #     scale=(0.98, 1.02),
    #     translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
    #     rotate=(-15, 15),  # rotate by -10 to +10 degrees
    # )),
])
'''