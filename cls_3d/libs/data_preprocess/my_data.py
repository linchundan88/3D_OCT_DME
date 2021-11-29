'''

write_csv_based_on_dir
  generate labels based on dir, and write images and labels to csv file


split_dataset   (split dataset into training and validation datasets)
split_dataset_cross_validation


write_csv_files() write list images and labels to csv file
  image_files and labels after split(based on pat_id or not), write to csv files.

'''

import os
import csv
import pandas as pd
import sklearn
import math
import random

#region image classification training

def write_csv_based_on_dir(filename_csv, base_dir, dict_mapping, match_type='header',
       list_file_ext=['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']):

    assert match_type in ['header', 'partial', 'end'], 'match type is error'

    if os.path.exists(filename_csv):
        os.remove(filename_csv)
    if not os.path.exists(os.path.dirname(filename_csv)):
        os.makedirs(os.path.dirname(filename_csv))
    if not base_dir.endswith('/'):
        base_dir = base_dir + '/'

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels'])

        for dir_path, _, files in os.walk(base_dir, False):
            for f in files:
                img_file_source = os.path.join(dir_path, f)
                (filedir, tempfilename) = os.path.split(img_file_source)
                (filename, extension) = os.path.splitext(tempfilename)
                if extension.upper() not in list_file_ext:
                    # print('file ext name:', f)
                    continue

                if not filedir.endswith('/'):
                    filedir += '/'

                for (k, v) in dict_mapping.items():
                    if match_type == 'header':
                        dir1 = os.path.join(base_dir, k)
                        if not dir1.endswith('/'):
                            dir1 += '/'

                        if dir1 in filedir:
                            csv_writer.writerow([img_file_source, v])
                            break
                    elif match_type == 'partial':
                        if '/' + k + '/' in filedir:
                            csv_writer.writerow([img_file_source, v])
                            break
                    elif match_type == 'end':
                        if filedir.endswith('/' + k + '/'):
                            csv_writer.writerow([img_file_source, v])
                            break


def split_dataset(filename_csv_or_df, valid_ratio=0.1, test_ratio=None,
                  shuffle=True, random_state=None):

    if isinstance(filename_csv_or_df, str):
        if filename_csv_or_df.endswith('.csv'):
            df = pd.read_csv(filename_csv_or_df)
        elif filename_csv_or_df.endswith('.xls') or filename_csv_or_df.endswith('.xlsx'):
            df = pd.read_excel(filename_csv_or_df)
    else:
        df = filename_csv_or_df

    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    if test_ratio is None:
        split_num = int(len(df)*(1-valid_ratio))
        data_train = df[:split_num]
        data_valid = df[split_num:]

        train_files = data_train['images'].tolist()
        train_labels = data_train['labels'].tolist()

        valid_files = data_valid['images'].tolist()
        valid_labels = data_valid['labels'].tolist()

        return train_files, train_labels, valid_files, valid_labels
    else:
        split_num_train = int(len(df) * (1 - valid_ratio - test_ratio))
        data_train = df[:split_num_train]
        split_num_valid = int(len(df) * (1 - test_ratio))
        data_valid = df[split_num_train:split_num_valid]
        data_test = df[split_num_valid:]

        train_files = data_train['images'].tolist()
        train_labels = data_train['labels'].tolist()

        valid_files = data_valid['images'].tolist()
        valid_labels = data_valid['labels'].tolist()

        test_files = data_test['images'].tolist()
        test_labels = data_test['labels'].tolist()

        return train_files, train_labels, valid_files, valid_labels, test_files, test_labels


def split_dataset_cross_validation(filename_csv_or_df, shuffle=True,
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


    train_files = [[] for _ in range(num_cross_validation)]
    train_labels = [[] for _ in range(num_cross_validation)]
    valid_files = [[] for _ in range(num_cross_validation)]
    valid_labels = [[] for _ in range(num_cross_validation)]

    batch_cross_validation = math.ceil(len(df) / num_cross_validation)

    for i in range(num_cross_validation):

        for j in range(len(df)):
            image_file = df.iat[i, 0]
            image_labels = df.iat[i, 1]

            if j >= i * batch_cross_validation and j<(i + 1) * batch_cross_validation:
                train_files[i].append(image_file)
                train_labels[i].append(image_labels)
            else:
                valid_files[i].append(image_file)
                valid_labels[i].append(image_labels)


        train_files[i] = train_files.tolist()
        train_labels[i] = train_labels.tolist()

        valid_files[i] = valid_files.tolist()
        valid_files[i] = valid_labels.tolist()

    return train_files, train_labels, valid_files, valid_labels



def write_images_labels_csv(files, labels, filename_csv):
    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    if not os.path.exists(os.path.dirname(filename_csv)):
        os.makedirs(os.path.dirname(filename_csv))

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels'])

        for i, file in enumerate(files):
            csv_writer.writerow([file, labels[i]])

    print('write csv ok!')




#endregion
