import os
import sys
sys.path.append(os.path.abspath('..'))
import pandas as pd
import csv
from libs.data_preprocess.my_data_patiend_id import split_dataset_by_pat_id

task_type_source = '3D_OCT_DME'
# ZEISS ,Topocon
dict_mapping = {'M1': 0, 'M2': 1}
dir_process = '/disk1/3D_OCT_DME/preprocess/128_128_128/'
data_version = 'v3'

csv_all_source = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type_source + '.csv'))
csv_train_source = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type_source + '_train.csv'))
csv_valid_source = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type_source + '_valid.csv'))
csv_test_source = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type_source + '_test.csv'))

task_type_dest = '3D_OCT_DME_M1_M2'
data_version = 'v3'
csv_all_target = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type_dest + '.csv'))
csv_train_target = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type_dest + '_train.csv'))
csv_valid_target = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type_dest + '_valid.csv'))
csv_test_target = os.path.abspath(os.path.join(os.path.abspath('..'),
            'datafiles', data_version, task_type_dest + '_test.csv'))


def write_csv(csv_source, csv_target):
    df = pd.read_csv(csv_source)
    with open(csv_target, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels'])

        for _, datarow in df.iterrows():
            img_file = datarow['images']
            if '/M1/' in img_file:
                csv_writer.writerow(['images', 0])
            if '/M2/' in img_file:
                csv_writer.writerow(['images', 1])

write_csv(csv_all_source, csv_all_target)
write_csv(csv_train_source, csv_train_target)
write_csv(csv_valid_source, csv_valid_target)
write_csv(csv_test_source, csv_test_target)


import pandas as pd
for csv_file in [csv_train_target, csv_valid_target, csv_test_target]:
    df = pd.read_csv(csv_file)
    print(len(df))
    for label in [0, 1]:
        df1 = df[df['labels'] == label]
        print(str(label), len(df1))

'''
1085
0 622
1 463
234
0 138
1 96
224
0 123
1 101
'''

print('OK')

