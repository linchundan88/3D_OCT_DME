import os

import cv2
import numpy as np


def save_npy(dir, depth_ratio=1, remainder=0, slice_num=128):

    for dir_path, subpaths, files in os.walk(dir, False):
        #zesis 1.jpeg, topocon 0.png
        if os.path.exists(os.path.join(dir_path, '1.jpeg')) \
                or os.path.exists(os.path.join(dir_path, '0.png')):

            list_images = []
            for i in range(slice_num):
                if i % depth_ratio == remainder:
                    if os.path.exists(os.path.join(dir_path, f'{str(i)}.png')):
                        img_file = os.path.join(dir_path,  f'{str(i)}.png')
                    elif os.path.exists(os.path.join(dir_path, f'{str(i+1)}.jpeg')):
                        img_file = os.path.join(dir_path, f'{str(i+1)}.jpeg')
                    else:
                        # raise Exception(f'file not found:{img_file}')
                        break

                    img1 = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) #(H,W)
                    list_images.append(img1)

            if len(list_images) != slice_num:  # some file error
                continue

            images = np.stack(list_images, axis=0)  #(H,W)->(D,H,W)

            pat_id = dir_path.split('/')[-1]
            if depth_ratio == 1 and remainder == 0:
                file_npy = os.path.join(dir_path, pat_id + '.npy')
            else:
                file_npy = os.path.join(dir_path, pat_id + f'_d{depth_ratio}_r{str(remainder)}.npy')
            print(file_npy)
            np.save(file_npy, images)