import math
import os
import random

import pandas as pd
import sklearn
import csv


def get_pat_id_oct(full_filename):
    _, filename = os.path.split(full_filename)
    filename = filename.replace('_d1_r0.npy', '')
    filename = filename.replace('_d2_r1.npy', '')
    filename = filename.replace('_d2_r0.npy', '')

    if len(filename.split('_')) > 0:
        # Topocon:02-000034_20160606_163238_OPT_L_001, 02-000034 patient id
        pat_id = filename.split('_')[0]
    else:
        if filename.startswith('00-') or filename.startswith('02-'):
            # Zesis:00-m00002-2od.npy patient id :00-m00002
            pat_id = filename.split('-')[0][0] + '_' + filename.split('-')[0][1]
        else:
            #ZESIS YPW19460508-od-2_d2_r0 pinyin + birthday day  patient id :YPW19460508
            pat_id = filename.split('-')[0]

    return pat_id

# 02-000003_20160601_120250_OPT_L_001.img.npy
def split_dataset_by_pat_id(filename_csv_or_df,
                            valid_ratio=0.1, test_ratio=None, shuffle=True, random_state=None):

    if isinstance(filename_csv_or_df, str):
        if filename_csv_or_df.endswith('.csv'):
            df = pd.read_csv(filename_csv_or_df)
        elif filename_csv_or_df.endswith('.xls') or filename_csv_or_df.endswith('.xlsx'):
            df = pd.read_excel(filename_csv_or_df)
    else:
        df = filename_csv_or_df

    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    list_patient_id = []
    for _, row in df.iterrows():
        filename = row['images']
        pat_id = get_pat_id_oct(filename)
        print(pat_id, filename)
        if pat_id not in list_patient_id:
            list_patient_id.append(pat_id)
    list_patient_id = sklearn.utils.shuffle(list_patient_id, random_state=random_state)

    if test_ratio is None:
        split_num = int(len(list_patient_id) * (1 - valid_ratio))
        list_patient_id_train = list_patient_id[:split_num]
        list_patient_id_valid = list_patient_id[split_num:]

        train_files, train_labels = [], []
        valid_files, valid_labels = [], []
        for _, row in df.iterrows():
            image_file, image_labels = row['images'], row['labels']
            pat_id = get_pat_id_oct(image_file)
            if pat_id in list_patient_id_train:
                train_files.append(image_file)
                train_labels.append(image_labels)
            if pat_id in list_patient_id_valid:
                valid_files.append(image_file)
                valid_labels.append(image_labels)

        return train_files, train_labels, valid_files, valid_labels

    else:
        split_num_train = int(len(list_patient_id) * (1 - valid_ratio - test_ratio))
        list_patient_id_train = list_patient_id[:split_num_train]
        split_num_valid = int(len(list_patient_id) * (1 - test_ratio))
        list_patient_id_valid = list_patient_id[split_num_train:split_num_valid]
        list_patient_id_test = list_patient_id[split_num_valid:]

        train_files, train_labels = [], []
        valid_files, valid_labels = [], []
        test_files, test_labels = [], []
        for _, row in df.iterrows():
            image_file, image_labels = row['images'], row['labels']
            _, filename = os.path.split(image_file)
            pat_id = get_pat_id_oct(filename)
            if pat_id in list_patient_id_train:
                train_files.append(image_file)
                train_labels.append(image_labels)
            if pat_id in list_patient_id_valid:
                valid_files.append(image_file)
                valid_labels.append(image_labels)
            if pat_id in list_patient_id_test:
                test_files.append(image_file)
                test_labels.append(image_labels)

        return train_files, train_labels, valid_files, valid_labels, test_files, test_labels



def get_list_patient_id(dir, del_sha1_header=False):
    list_patient_id = []

    for dir_path, _, files in os.walk(dir, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)
            if not '/original/' in img_file_source:
                continue

            _, filename = os.path.split(img_file_source)
            file_basename, file_extension = os.path.splitext(filename)

            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF', '.PPF']:
                # print('file ext name:', f)
                continue

            if del_sha1_header:
                # 0a1d0ef72e0bc69c01f87502faba2524fbd7d7b6#a9c63c7b-32f7-4a41-bc94-03469fd8c7e0.14.jpg
                file_basename = file_basename.split('#')[1]

            # '0f104b1b-d1cd-4acc-8a1b-4e88c61d18b0.16.jpg'
            patient_id = file_basename.split('.')[0]
            if patient_id not in list_patient_id:
                list_patient_id.append(patient_id)

    return list_patient_id


def split_list_patient_id(list_patient_id, valid_ratio=0.1, test_ratio=0.1,
                          random_seed=18888):
    random.seed(random_seed)
    random.shuffle(list_patient_id)

    if test_ratio is None:
        split_num = int(len(list_patient_id) * (1 - valid_ratio))
        list_patient_id_train = list_patient_id[:split_num]
        list_patient_id_valid = list_patient_id[split_num:]

        return list_patient_id_train, list_patient_id_valid
    else:
        split_num_train = int(len(list_patient_id) * (1 - valid_ratio - test_ratio))
        list_patient_id_train = list_patient_id[:split_num_train]
        split_num_valid = int(len(list_patient_id) * (1 - test_ratio))
        list_patient_id_valid = list_patient_id[split_num_train:split_num_valid]
        list_patient_id_test = list_patient_id[split_num_valid:]

        return list_patient_id_train, list_patient_id_valid, list_patient_id_test



def split_dataset_by_pat_id_cross_validation(filename_csv_or_df, shuffle=True,
             num_cross_validation=5, random_state=None):

    if isinstance(filename_csv_or_df, str):
        if filename_csv_or_df.endswith('.csv'):
            df = pd.read_csv(filename_csv_or_df)
        elif filename_csv_or_df.endswith('.xls') or filename_csv_or_df.endswith('.xlsx'):
            df = pd.read_excel(filename_csv_or_df)
    else:
        df = filename_csv_or_df

    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    list_patient_id = []
    for _, row in df.iterrows():
        image_file = row['images']
        _, filename = os.path.split(image_file)
        pat_id = filename.split('.')[0]
        if pat_id not in list_patient_id:
            list_patient_id.append(pat_id)

    train_files = [[] for _ in range(num_cross_validation)]
    train_labels = [[] for _ in range(num_cross_validation)]
    valid_files = [[] for _ in range(num_cross_validation)]
    valid_labels = [[] for _ in range(num_cross_validation)]

    for i in range(num_cross_validation):
        patient_id_batch = math.ceil(len(list_patient_id) / num_cross_validation)

        list_patient_id_valid = list_patient_id[i*patient_id_batch: (i+1)*patient_id_batch]

        list_patient_id_train = []
        for pat_id in list_patient_id:
            if pat_id not in list_patient_id_valid:
                list_patient_id_train.append(pat_id)

        for _, row in df.iterrows():
            image_file = row['images']
            image_labels = row['labels']
            _, filename = os.path.split(image_file)

            pat_id = filename.split('.')[0]

            if pat_id in list_patient_id_train:
                train_files[i].append(image_file)
                train_labels[i].append(image_labels)

            if pat_id in list_patient_id_valid:
                valid_files[i].append(image_file)
                valid_labels[i].append(image_labels)

    return train_files, train_labels, valid_files, valid_labels