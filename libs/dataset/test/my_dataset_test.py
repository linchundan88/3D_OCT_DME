import cv2
from imgaug import augmenters as iaa
from libs.dataset.my_dataset import Dataset_CSV_3d
import numpy as np
import os

sometimes = lambda aug: iaa.Sometimes(0.96, aug)
imgaug = iaa.Sequential([
      # iaa.CropAndPad(percent=(-0.04, 0.04)),
      iaa.Fliplr(0.5),  # horizontally flip 50% of the images
      # iaa.Flipud(0.2),  # horizontally flip 50% of the images

      iaa.GaussianBlur(sigma=(0.0, 0.3)),
      iaa.MultiplyBrightness(mul=(0.8, 1.2)),
      sometimes(iaa.ContrastNormalization((0.9, 1.1))),

      # iaa.Sometimes(0.9, iaa.Add((-6, 6))),
      # sometimes(iaa.Affine(
      #     scale=(0.98, 1.02),
      #     translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
      #     rotate=(-15, 15),  # rotate by -10 to +10 degrees
      # )),
])

csv_file = '3D_OCT_DME.csv'

aug_times = 10

for aug_time in range(aug_times):
      dataset1 = Dataset_CSV_3d(csv_file=csv_file,
            image_shape=(64, 64), channel_first=False, test_mode=False, imgaug_iaa=imgaug
      )

      for (x, y) in dataset1:
            tensor_3d = x.numpy()  # (D,H,W,C)
            tensor_3d *= 255.
            for i in range(tensor_3d.shape[0]):
                  image1 = tensor_3d[i, :, :, 0]
                  os.makedirs(f'/tmp2/{aug_time}/0/', exist_ok=True)
                  cv2.imwrite(f'/tmp2/{aug_time}/0/{i}.jpg', image1)
            for i in range(tensor_3d.shape[1]):
                  image1 = tensor_3d[:, i, :, 0]
                  os.makedirs(f'/tmp2/{aug_time}/1/', exist_ok=True)
                  cv2.imwrite(f'/tmp2/{aug_time}/1/{i}.jpg', image1)
            for i in range(tensor_3d.shape[2]):
                  image1 = tensor_3d[:, :, i, 0]
                  os.makedirs(f'/tmp2/{aug_time}/2/', exist_ok=True)
                  cv2.imwrite(f'/tmp2/{aug_time}/2/{i}.jpg', image1)

            break


print('ok')