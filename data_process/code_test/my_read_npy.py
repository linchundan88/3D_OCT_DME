import os
import numpy as np
import cv2

npy_file = '/disk1/3D_OCT_DME/preprocess/128_128_128/Topocon/Topocon/无黄斑水肿/02-000043_20160610_095207_OPT_R_001/02-000043_20160610_095207_OPT_R_001_d0_r0.npy'


tensor_3d = np.load(npy_file)
dir_dest = '/tmp2/aaa'
os.makedirs(dir_dest, exist_ok=True)

for i in range(tensor_3d.shape[0]):
    image = tensor_3d[i]
    file_dest = os.path.join(dir_dest, f'{i}.png')

    cv2.imwrite(file_dest, image)

print('OK')