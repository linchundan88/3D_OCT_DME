import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
sys.path.append(os.path.abspath('../..'))
from torch.utils.data import DataLoader
import pandas as pd
from libs.dataset.my_dataset_torchio import Dataset_CSV_test
from libs.neural_networks.helper.my_predict_multi_class import predict_single_model
import shutil
from libs.neural_networks.model.my_get_model import get_model

filename_csv = os.path.join(os.path.abspath('../..'),
                'datafiles', 'v3', f'3D_OCT_DME.csv')
dir_original = '/disk1/3D_OCT_DME/original/'
dir_preprocess = '/disk1/3D_OCT_DME/preprocess/128_128_128/'
dir_dest = '/tmp2/3D_OCT_DME/3D_OCT_DME_confusion_files_2021_6_6/'
export_confusion_files = False

num_class = 2
model_name = 'cls_3d'
model_file = os.path.join(os.path.abspath('../..'), 'trained_models', 'multi_class', 'cls_3d.pth')
image_shape = (64, 64)

model = get_model(model_name, num_class=num_class, model_file=model_file)


ds_test = Dataset_CSV_test(csv_file=filename_csv, image_shape=image_shape,
                           depth_start=0, depth_interval=2, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=32, pin_memory=True, num_workers=4)

(probs, labels_pd) = predict_single_model(model, loader_test, activation='softmax')

df = pd.read_csv(filename_csv)
image_files, labels_gt = list(df['images']), list(df['labels'])
from sklearn.metrics import confusion_matrix
print(confusion_matrix(labels_gt, labels_pd))


if export_confusion_files:
    for image_file, label_gt, prob in zip(image_files, labels_gt, probs):
        label_predict = prob.argmax()
        if label_gt != label_predict:
            dir_base = os.path.dirname(image_file.replace(dir_preprocess, dir_original))
            for dir_path, _, files in os.walk(dir_base, False):
                for f in files:
                    file_full_path = os.path.join(dir_path, f)
                    _, file_name = os.path.split(file_full_path)
                    _, file_ext = os.path.splitext(file_name)
                    if file_ext.lower() in ['.jpeg', '.jpg', '.png']:
                        file_partial_path = file_full_path.replace(dir_original, '')
                        file_name1 = os.path.join(dir_dest, f'{label_gt}_{label_predict}', file_partial_path)
                        tmp_dir, tmp_filename = os.path.split(file_name1)
                        if label_gt == 0:
                            dir_prob = f'prob{int(prob[0] * 100)}_'
                        if label_gt == 1:
                            dir_prob = f'prob{int(prob[1] * 100)}_'

                        list_tmp_dir = tmp_dir.split('/')
                        list_tmp_dir[-1] = dir_prob + list_tmp_dir[-1]
                        img_dest = os.path.join('/'.join(list_tmp_dir), tmp_filename)
                        print(img_dest)
                        os.makedirs(os.path.dirname(img_dest), exist_ok=True)
                        shutil.copy(file_full_path, img_dest)

    print('export confusion files ok!')
