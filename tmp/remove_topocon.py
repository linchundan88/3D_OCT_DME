import os
import shutil

dir1 = '/disk1/3D_OCT_DME/Topocon_dicom/'

for dir_path, subpaths, files in os.walk(dir1, False):
    for f in files:
        image_file_source = os.path.join(dir_path, f)
        file_base, file_ext = os.path.splitext(image_file_source)  # 分离文件名与扩展名
        if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            continue

        print(f'remove file:{image_file_source}')
        os.remove(image_file_source)

print('OK')