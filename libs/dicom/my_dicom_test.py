import os
import pydicom
import numpy as np
import cv2


dicom_file = '/tmp7/b/02-000220_20160830_114304_OPT_L_001.dcm'

ds2 = pydicom.dcmread(dicom_file)
array1 = ds2.pixel_array

for i in range(array1.shape[0]):
    img = array1[i]
    img_save = f'/tmp7/c/{i}.jpg'
    cv2.imwrite(img_save, img)

print('OK')

