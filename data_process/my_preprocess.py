
import os
import sys
sys.path.append(os.path.abspath('..'))
from libs.dicom.my_dicom import dicom_save_dirs, slices_to_npy
from libs.img_preprocess.my_image_helper import resize_images



DICOM_TO_IMAGES = False

RESIZE_IMAGES_TOPOCON = False
RESIZE_IMAGES_ZEISS = True


if DICOM_TO_IMAGES:
    source_dir_dicom = '/disk1/3D_OCT_DME/Topocon_dicom/'
    dir_dest = '/disk1/3D_OCT_DME/original/Topocon/'
    for label_str in ['M0', 'M1', 'M2']:
        dir_source_tmp = os.path.join(source_dir_dicom, label_str)
        dir_dest_tmp = os.path.join(dir_dest, label_str)
        dicom_save_dirs(dir_source_tmp, dir_dest_tmp, save_npy=True, save_image_files=True)



image_shape = (128, 128)
if RESIZE_IMAGES_TOPOCON:
    dir_source = '/disk1/3D_OCT_DME/original/Topocon/'
    dir_dest = '/disk1/3D_OCT_DME/preprocess_128_128_128/Topocon/'
    for label_str in ['M0', 'M1', 'M2']:
        dir_source_tmp = os.path.join(dir_source, label_str)
        dir_dest_tmp = os.path.join(dir_dest, label_str)
        resize_images(dir_source_tmp, dir_dest_tmp, p_image_to_square=True, image_shape=image_shape)
        slices_to_npy(dir_dest_tmp)

if RESIZE_IMAGES_ZEISS:
    dir_source = '/disk1/3D_OCT_DME/original/ZEISS/'
    dir_dest = '/disk1/3D_OCT_DME/preprocess_128_128_128/ZEISS/'
    for label_str in ['M0', 'M1', 'M2']:
        dir_source_tmp = os.path.join(dir_source, label_str)
        dir_dest_tmp = os.path.join(dir_dest, label_str)
        resize_images(dir_source_tmp, dir_dest_tmp, p_image_to_square=True, image_shape=image_shape)
        slices_to_npy(dir_dest_tmp)


print('OK')