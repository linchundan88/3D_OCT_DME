
import os
import sys
sys.path.append(os.path.abspath('../..'))
from cls_3d.libs.img_preprocess.my_image_helper import resize_images
from cls_3d.libs.dicom.my_dicom import slices_to_npy

if __name__ == '__main__':
    # dir_source = '/disk1/3D_OCT_DME/original/ZEISS/'
    dir_source = '/disk1/3D_OCT_DME/original/ZEISS/M0_no_diabetes/'

    '''
    dest_dir = '/disk1/3D_OCT_DME/preprocess/64_64_64/ZEISS/'
    resize_images_dir(source_dir, dest_dir,
                      p_image_to_square=True, image_shape=(64, 64))
    # save_npy(dest_dir, depth_ratio=1, remainder=0)
    save_npy(dest_dir, depth_ratio=2, remainder=0)
    save_npy(dest_dir, depth_ratio=2, remainder=1)
    '''

    # dir_dest = '/disk1/3D_OCT_DME/preprocess/128_128_128/ZEISS'
    dir_dest = '/disk1/3D_OCT_DME/preprocess/128_128_128/ZEISS/M0_no_diabetes'
    resize_images(dir_source, dir_dest,
                  p_image_to_square=True, image_shape=(128, 128))
    slices_to_npy(dir_dest)

    print('OK')



