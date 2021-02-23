
import os
from libs.dicom.my_dicom import dicom_save_dirs

source_dir = '/disk1/3D_OCT_DME/Topocon_dicom/'

#region DICOM->images
# dir_dest = '/disk1/3D_OCT_DME/original/Topocon/'
# dicom_save_dirs(source_dir, dir_dest, save_image_files=True)
#endregion

'''
#region resize to 64_64_64
image_shape = (64, 64)
dir_dest = '/disk1/3D_OCT_DME/preprocess/64_64_64/Topocon/'
z_interval = 2
for dir_str in ['Topocon', 'Topocon_2019_1_19']:
    for label_str in ['无黄斑水肿', '未累及中央的黄斑水肿', '累及中央的黄斑水肿']:
        source_dir_tmp = os.path.join(source_dir, dir_str, label_str)
        dir_dest_tmp = os.path.join(dir_dest, dir_str, label_str)

        dicom_save_dirs(source_dir_tmp, dir_dest_tmp,
                        image_shape=image_shape, save_npy=True, save_image_files=True,
                        depth_interval=z_interval, remainder=0)
        dicom_save_dirs(source_dir_tmp, dir_dest_tmp,
                        image_shape=image_shape, save_npy=True, save_image_files=False,
                        depth_interval=z_interval, remainder=1)

#endregion
'''



#region 128_128_128

image_shape = (128, 128)
dir_dest = '/disk1/3D_OCT_DME/preprocess/128_128_128/Topocon/'
for dir_str in ['Topocon', 'Topocon_2019_1_19']:
    for label_str in ['无黄斑水肿', '未累及中央的黄斑水肿', '累及中央的黄斑水肿']:
        source_dir_tmp = os.path.join(source_dir, dir_str, label_str)
        dir_dest_tmp = os.path.join(dir_dest, dir_str, label_str)

        dicom_save_dirs(source_dir_tmp, dir_dest_tmp,
                        image_shape=image_shape, save_npy=True, save_image_files=True)


#endregion



print('OK')