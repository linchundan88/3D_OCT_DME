
import numpy as np
import random
import torchio as tio
from torch.utils.data import DataLoader
import cv2
import os

file_npy = '/disk1/3D_OCT_DME/preprocess/128_128_128/Topocon/Topocon/无黄斑水肿/02-000037_20160607_101020_OPT_R_001/02-000037_20160607_101020_OPT_R_001.npy'
resample_ratio = (1, 1, 1)
crop_pad_pixel = 15
crop_pad_ratio = (3, 9)
image_shape = (128, 128)
imgaug_iaa = None
channel_first = True


array_3d = np.load(file_npy)  # shape (D,H,W)
if array_3d.ndim > 3:
    array_3d = np.squeeze(array_3d)

int_r = random.randint(0, 20)
if int_r % 2 == 0:
    array_3d = array_3d[0::2]
else:
    array_3d = array_3d[1::2]

subject1 = tio.Subject(
    oct=tio.ScalarImage(tensor=np.expand_dims(array_3d, axis=0)),
)
subjects_list = [subject1]

transform_0 = tio.Compose([tio.Resample(resample_ratio)])

crop_a = random.randint(crop_pad_ratio[0], crop_pad_ratio[1])
crop_b = crop_pad_pixel - crop_a
pad_a = random.randint(crop_pad_ratio[0], crop_pad_ratio[1])
pad_b = crop_pad_pixel - pad_a
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
    tio.Resample(resample_ratio),
    # tio.RescaleIntensity((0, 255))
])

if random.randint(1, 10) == 5:
    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform_0)
else:
    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform_1)
loader1 = DataLoader(subjects_dataset, batch_size=1, num_workers=0)

for subjects_batch in loader1:
    inputs = subjects_batch['oct'][tio.DATA]
    array1 = inputs.cpu().numpy()
    # array1.shape (B,C,D,H,W)  (1, 1, 64, 128, 128)
    array1 = np.squeeze(array1[0], axis=0)  # array1.shape: (D,H,W)
    array1 = array1.astype(np.uint8)

if imgaug_iaa is not None:
    imgaug_iaa.deterministic = True
else:
    if (array1.shape[1:3]) == (image_shape[0:2]):  # (H,W)
        array2 = np.expand_dims(array_3d, axis=-1)  # (D,H,W,C)

if 'array2' not in locals().keys():
    list_images = []
    for i in range(array1.shape[0]):  # D,H,W
        img = array1[i, :, :]
        if (img.shape[0:2]) != (image_shape[0:2]):  # (H,W)
            img = cv2.resize(img, (image_shape[1], image_shape[0]))  # resize(width,height)

        # cvtColor do not support float64
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2BGR)
        # other wise , MultiplyBrightness error
        img = img.astype(np.uint8)
        if imgaug_iaa is not None:
            img = imgaug_iaa(image=img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        list_images.append(img)

    array2 = np.array(list_images)  # (D,H,W)
    array2 = np.expand_dims(array2, axis=-1)  # (D,H,W,C)

if imgaug_iaa is not None:
    imgaug_iaa.deterministic = False

if channel_first:
    array2 = np.transpose(array2, (3, 0, 1, 2))

array2 = array2.astype(np.float32)
array2 = array2 / 255.

tensor_3d = np.squeeze(array2)
tensor_3d *= 255.
for j in range(tensor_3d.shape[0]):
    image1 = tensor_3d[j]
    file_img = os.path.join(dir_dest,  f'{j}.png')
    os.makedirs(os.path.dirname(file_img), exist_ok=True)
    # print(file_img)
    cv2.imwrite(file_img, image1)


print('OK')