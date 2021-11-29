import os
import sys
sys.path.append(os.path.abspath('../..'))
from cls_3d.libs.data_preprocess.my_data import write_images_labels_csv, write_csv_based_on_dir
from cls_3d.libs.data_preprocess.my_data_patiend_id import split_dataset_by_pat_id



task_type = '3D_OCT_DME_M0_M1_M2'
dict_mapping = {'M0': 0, 'M0_no_diabetes': 0, 'M1': 1, 'M2': 2}
dir_process = '/disk1/3D_OCT_DME/preprocess/128_128_128/'
data_version = 'v3'

csv_all = os.path.abspath(os.path.join(os.path.abspath('../'),
            'datafiles', data_version, task_type + '.csv'))
write_csv_based_on_dir(csv_all, dir_process, dict_mapping,
                       match_type='partial', list_file_ext=['.npy', '.NPY'])
files_train, labels_train, files_valid, labels_valid, files_test, labels_test = \
    split_dataset_by_pat_id(csv_all,
                            valid_ratio=0.15, test_ratio=0.15, random_state=111)
csv_train = os.path.abspath(os.path.join(os.path.abspath('../'),
            'datafiles', data_version, task_type + '_train.csv'))
csv_valid = os.path.abspath(os.path.join(os.path.abspath('../'),
            'datafiles', data_version, task_type + '_valid.csv'))
csv_test = os.path.abspath(os.path.join(os.path.abspath('../'),
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
    for label in [0, 1, 2]:
        df1 = df[df['labels'] == label]
        print(str(label), len(df1))

''' 
4789
0 3704
1 622
2 463
1026
0 792
1 138
2 96
1041
0 817
1 123
2 101
'''




print('OK')

