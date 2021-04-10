import os
import shutil

dir1 = '/disk1/3D_OCT_DME/original/Topocon/3D_OCT_DME_confusion_files_2021_2_17'
dir2 = '/disk1/3D_OCT_DME/Topocon_dicom/'

# 'M0_ M1_ M2_
for dir_path, subpaths, files in os.walk(dir1, False):
    filename_base = dir_path.split('/')[-1]
    if filename_base.startswith('M0-'):
        label_gt = 'M0'
    elif filename_base.startswith('M1-'):
        label_gt = 'M1'
    elif filename_base.startswith('M2-'):
        label_gt = 'M2'
    else:
        continue

    filename_base = filename_base[3:]
    dest_file = os.path.join(dir2, label_gt, filename_base + '.dicom')

    if os.path.exists(os.path.join(dir2, 'M0',  filename_base+'.dcm')):
        source_file = os.path.join(dir2, 'M0',  filename_base+'.dcm')
    elif os.path.exists(os.path.join(dir2, 'M1',  filename_base+'.dcm')):
        source_file = os.path.join(dir2, 'M1',  filename_base+'.dcm')
    elif os.path.exists(os.path.join(dir2, 'M2',  filename_base+'.dcm')):
        source_file = os.path.join(dir2, 'M2',  filename_base+'.dcm')
    else:
        continue

    if source_file != dest_file:
        print(f'{source_file} to:{dest_file}')
        shutil.move(source_file, dest_file)

print('OK')

