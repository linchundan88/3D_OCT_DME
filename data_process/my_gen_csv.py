import os
import sys
sys.path.append(os.path.abspath('..'))
from libs.data_preprocess.my_data import write_images_labels_csv, write_csv_based_on_dir
from libs.data_preprocess.my_data_patiend_id import split_dataset_by_pat_id

task_type = '3D_OCT_DME'
#zeiss  ZEISS ,Topocon
dict_mapping = {'M0': 0, 'M0_no_diabetes': 0, 'M1': 1, 'M2': 1}
dir_process = '/disk1/3D_OCT_DME/preprocess/128_128_128/'
data_version = 'v2'

csv_all = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type + '.csv'))
write_csv_based_on_dir(csv_all, dir_process, dict_mapping,
                       match_type='partial', list_file_ext=['.npy', '.NPY'])


# 02-000003_20160601_120250_OPT_L_001.img.npy
files_train, labels_train, files_valid, labels_valid, files_test, labels_test = \
    split_dataset_by_pat_id(csv_all,
                            valid_ratio=0.15, test_ratio=0.15, random_state=111)

csv_train = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type + '_train.csv'))
csv_valid = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type + '_valid.csv'))
csv_test = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type + '_test.csv'))

write_images_labels_csv(files_train, labels_train,
                        filename_csv=csv_train)
write_images_labels_csv(files_valid, labels_valid,
                        filename_csv=csv_valid)
write_images_labels_csv(files_test, labels_test,
                        filename_csv=csv_test)

import pandas as pd
for csv_file in [csv_train, csv_valid, csv_test]:
    df = pd.read_csv(csv_file)
    print(len(df))
    for label in [0, 1]:
        df1 = df[df['labels'] == label]
        print(str(label), len(df1))

''' 

v1_topocon_128_128_128
3349
0 2962
1 387
722
0 639
1 83
706
0 638
1 68

v2 topocon and ZEISS
4414
0 3445
1 969
952
0 745
1 207
985
0 766
1 219


4814
0 3831
1 983
1021
0 815
1 206
1022
0 816
1 206


'''


#region no diabetes
'''
dict_mapping = {'M0_no_diabetes': 0}
dir_process = '/disk1/3D_OCT_DME/preprocess/128_128_128/ZEISS/M0_no_diabetes'
csv_M0_no_diabetes = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type + '_no_diabetes.csv'))
write_csv_based_on_dir(csv_M0_no_diabetes, dir_process, dict_mapping,
                       match_type='partial', list_file_ext=['.npy', '.NPY'])
'''
#endregion


print('OK')

