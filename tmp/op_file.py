import os
import shutil

dir1 = '/disk1/3D_OCT_DME/original/Topocon/3D_OCT_DME_confusion_files_2021_2_17'
dir2 = '/disk1/3D_OCT_DME/original/Topocon/Topocon'

def del_dir(dir1, dir_name_last):
    for dir_path, _, files in os.walk(dir1, False):
        if dir_name_last == dir_path.split('/')[-1]:
            print(f'del dir:{dir_path}')
            shutil.rmtree(dir_path)

#
# 'M0_ M1_ M2_
for dir_path, _, files in os.walk(dir1, False):
    dir_name_last = dir_path.split('/')[-1]
    if dir_name_last.startswith('M0-'):
        label_gt = 'M0'
    elif dir_name_last.startswith('M1-'):
        label_gt = 'M1'
    elif dir_name_last.startswith('M2-'):
        label_gt = 'M2'
    else:
        continue

    print(f'current dir:{dir_name_last}')

    dir_name_last = dir_name_last[3:]
    del_dir(dir2, dir_name_last)  #delete old duplicate files

    dest_dir = os.path.join(dir2, label_gt, dir_name_last)
    # os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    print(f'copy dir:{dest_dir}')
    shutil.copytree(dir_path, dest_dir)



print('OK')

