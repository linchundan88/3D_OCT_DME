from libs.img_preprocess.my_image_helper import resize_images_dir
from libs.oct.my_oct_helper import save_npy


if __name__ == '__main__':
    source_dir = '/disk1/3D_OCT_DME/original/ZEISS/'

    '''
    dest_dir = '/disk1/3D_OCT_DME/preprocess/64_64_64/ZEISS/'
    resize_images_dir(source_dir, dest_dir,
                      p_image_to_square=True, image_shape=(64, 64))
    # save_npy(dest_dir, depth_ratio=1, remainder=0)
    save_npy(dest_dir, depth_ratio=2, remainder=0)
    save_npy(dest_dir, depth_ratio=2, remainder=1)
    '''

    dest_dir = '/disk1/3D_OCT_DME/preprocess/128_128_128/ZEISS'
    resize_images_dir(source_dir, dest_dir,
                      p_image_to_square=True, image_shape=(128, 128))
    save_npy(dest_dir)

    print('OK')



