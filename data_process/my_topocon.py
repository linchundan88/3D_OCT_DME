
import os
from libs.dicom.my_dicom import dicom_save_dirs

source_dir_dicom = '/disk1/3D_OCT_DME/Topocon_dicom/'

#region DICOM->images
# dir_dest = '/disk1/3D_OCT_DME/original/Topocon/'
# dicom_save_dirs(source_dir_dicom, dir_dest, save_image_files=True)
#endregion


#region preprocessing 128_128_128

image_shape = (128, 128)
dir_dest = '/disk1/3D_OCT_DME/preprocess/128_128_128/Topocon/'

for label_str in ['M0', 'M1', 'M2']:
    dir_source_tmp = os.path.join(source_dir_dicom, label_str)
    dir_dest_tmp = os.path.join(dir_dest, label_str)

    dicom_save_dirs(dir_source_tmp, dir_dest_tmp,
                    image_shape=image_shape, save_npy=True, save_image_files=True)

#endregion



print('OK')