
import os
import numpy as np
import torchio as tio
from torch.utils.data import DataLoader
import cv2
import random


def aug_tensors(file_npy, dest_dir, resample_ratio=(1, 1, 1),
                save_images=False, aug_times=10):
    array_npy = np.load(file_npy)

    for aug_time in range(aug_times):
        int_r = random.randint(0, 20)
        if int_r % 2 == 0:
            array_4d = np.expand_dims(array_npy[0::2, :, :], axis=0)
        else:
            array_4d = np.expand_dims(array_npy[1::2, :, :], axis=0)

        subject1 = tio.Subject(
            oct=tio.ScalarImage(tensor=array_4d),
        )

        subjects_list = [subject1]

        crop_a = random.randint(3, 9)
        crop_b = 15 - crop_a
        pad_a = random.randint(4, 8)
        pad_b = 15 - pad_a

        if aug_time == 0:
            transform = tio.Compose([tio.Resample(resample_ratio)])
        else:
            transform = tio.Compose([
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
                tio.Crop(cropping=(0, 0, crop_a, crop_b, 0, 0)),  #(d,h,w) crop height
                tio.Pad(padding=(0, 0, pad_a, pad_b, 0, 0)),
                tio.Resample(resample_ratio),
                # tio.RescaleIntensity((0, 255))
                ]
            )

        subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)

        #subjects_dataset[0].shape (C,D,H,W)
        inputs = subjects_dataset[0]['oct'][tio.DATA]

        array_3d = np.squeeze(inputs.cpu().numpy(), axis=0)  # (D,H,W)
        array_3d = array_3d.astype(np.uint8)

        if save_images:
            for index_slice in range(array_3d.shape[0]):
                image = array_3d[index_slice]
                file_dest = os.path.join(dest_dir, str(aug_time), f'{index_slice}.png')
                os.makedirs(os.path.dirname(file_dest), exist_ok=True)
                cv2.imwrite(file_dest, image)

        file_npy_dest = os.path.join(dest_dir, f'aug_{aug_time}.npy')
        os.makedirs(os.path.dirname(file_npy_dest), exist_ok=True)
        np.save(file_npy_dest, array_3d)


aug_tensors('/disk1/3D_OCT_DME/preprocess/128_128_128/Topocon/Topocon/累及中央的黄斑水肿/02-000003_20160601_120201_OPT_R_001/02-000003_20160601_120201_OP'
            'T_R_001.npy', '/tmp2/bb', aug_times=10, save_images=True)

'''
source_dir = '/disk1/3D_OCT_DME/preprocess/128_128_128/Topocon/'
for dir_path, subpaths, files in os.walk(source_dir, False):
    for f in files:
        file_npy = os.path.join(dir_path, f)
        file_dir, filename = os.path.split(file_npy)
        file_base, file_ext = os.path.splitext(filename)
        if file_ext.lower() not in ['.npy']:
            continue

        aug_times(file_npy, '/tmp2/bbb')
        exit

'''

print('ok')