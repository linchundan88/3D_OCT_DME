import cv2
from imgaug import augmenters as iaa
from libs.dataset.my_dataset_torchio import Dataset_CSV
import numpy as np
import os


csv_test = 'test.csv'
image_shape = (128, 128)
dir_dest = '/tmp2/cc/aug'

sometimes = lambda aug: iaa.Sometimes(0.96, aug)
imgaug = iaa.Sequential([
      # iaa.CropAndPad(percent=(-0.04, 0.04)),
      # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
      # iaa.Flipud(0.2),  # horizontally flip 50% of the images

      # iaa.GaussianBlur(sigma=(0.0, 0.3)),
      # iaa.MultiplyBrightness(mul=(0.8, 1.2)),
      # sometimes(iaa.ContrastNormalization((0.9, 1.1))),

      iaa.Sometimes(0.9, iaa.Add((-6, 6))),
      # sometimes(iaa.Affine(
      #     scale=(0.98, 1.02),
      #     translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
      #     rotate=(-15, 15),  # rotate by -10 to +10 degrees
      # )),
])

aug_times = 10
for aug_time in range(aug_times):
      ds_test = Dataset_CSV(csv_file=csv_test, image_shape=image_shape, test_mode=True)

      tensor_3d = ds_test[0].numpy()  # (B,C,D,H,W) -> (D,H,W,C)
      tensor_3d = np.squeeze(tensor_3d)
      tensor_3d *= 255.
      for j in range(tensor_3d.shape[0]):
            image1 = tensor_3d[j]
            file_img = os.path.join(dir_dest, str(aug_time), f'{j}.png')
            os.makedirs(os.path.dirname(file_img), exist_ok=True)
            # print(file_img)
            cv2.imwrite(file_img, image1)

print('OK')