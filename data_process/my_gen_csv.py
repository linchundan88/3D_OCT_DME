import os
from libs.data_preprocess import my_data
from libs.data_preprocess.my_data_patiend_id import split_dataset_by_pat_id

train_type = '3D_OCT_DME'
# dict_mapping = {'无黄斑水肿': 0, '未累及中央的黄斑水肿': 1, '累及中央的黄斑水肿': 1,
#                 'M0': 0, 'M1': 1, 'M2': 1}
# dir_original = '/disk1/3D_OCT_DME/preprocess/64_64_64'
# data_version = 'v1_topocon_zeiss_64_64_64'
dict_mapping = {'无黄斑水肿': 0, '未累及中央的黄斑水肿': 1, '累及中央的黄斑水肿': 1}
dir_original = '/disk1/3D_OCT_DME/preprocess/64_64_64/Topocon'
data_version = 'v1_topocon_64_64_64'

dir_original = '/disk1/3D_OCT_DME/preprocess/128_128_128/Topocon'
data_version = 'v1_topocon_128_128_128'

filename_csv = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, train_type + '.csv'))
my_data.write_csv_based_on_dir(filename_csv, dir_original, dict_mapping,
    match_type='partial', list_file_ext=['.npy', '.NPY'])

# 02-000003_20160601_120250_OPT_L_001.img.npy
train_files, train_labels, valid_files, valid_labels, test_files, test_labels = \
    split_dataset_by_pat_id(filename_csv, match_type='OCT',
                            valid_ratio=0.15, test_ratio=0.15, random_state=456)

filename_csv_train = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, train_type + '_split_patid_train.csv'))
filename_csv_valid = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, train_type + '_split_patid_valid.csv'))
filename_csv_test = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, train_type + '_split_patid_test.csv'))

my_data.write_images_labels_csv(train_files, train_labels,
                    filename_csv=filename_csv_train)
my_data.write_images_labels_csv(valid_files, valid_labels,
                    filename_csv=filename_csv_valid)
my_data.write_images_labels_csv(test_files, test_labels,
                    filename_csv=filename_csv_test)

import pandas as pd
for file_csv in [filename_csv_train, filename_csv_valid, filename_csv_test]:
    df = pd.read_csv(file_csv)
    print(len(df))
    for label in [0, 1]:
        df1 = df[df['labels'] == label]
        print(str(label), len(df1))

''' 
v1_topocon_64_64_64:
6708
0 5958
1 750
1420
0 1254
1 166
1426
0 1266
1 160

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
OK

v1_topocon_zeiss:
7821
0 6446
1 1375
1685
0 1406
1 279
1692
0 1442
1 250

'''

print('OK')

