import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from libs.dataset.my_dataset_torchio import Dataset_CSV_test
from libs.neural_networks.helper.my_predict_binary_class import predict_multiple_models
from libs.neural_networks.model.my_get_model import get_model
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1')
parser.add_argument('--task_type', default='3D_OCT_DME_M0_M1M2_test')  #3D_OCT_DME_M0_M1M2, 3D_OCT_DME_M1_M2
parser.add_argument('--image_shape', nargs='+', type=int, default=(64, 64))
parser.add_argument('--data_version', default='v1')
parser.add_argument('--dir_dest', default='/tmp2/3D_OCT_DME/2022_4_28_64_64/m0_m1m2')
parser.add_argument('--export_confusion_files', action='store_true')
args = parser.parse_args()

csv_file = Path(__file__).resolve().parent.parent.parent / 'datafiles' / args.data_version / f'{args.task_type}.csv'
dir_original = '/disk1/3D_OCT_DME/original/'
dir_preprocess = '/disk1/3D_OCT_DME/preprocess_128_128_128/'

threshold = 0.5

models_dicts = []

path_model_base = Path(__file__).resolve().parent.parent.parent / 'trained_models' / '2022_4_28_64_64'
if args.task_type.startswith('3D_OCT_DME_M0_M1M2'):
    path_model = path_model_base / 'binary_class_m0_m1m2'
if args.task_type.startswith('3D_OCT_DME_M1_M2'):
    path_model = path_model_base / 'binary_class_m1_m2'

model_name = 'cls_3d'
model_file = path_model / 'cls_3d.pth'
model = get_model(model_name, 1, model_file=model_file)
ds_test = Dataset_CSV_test(csv_file=csv_file, image_shape=args.image_shape,
                           depth_start=0, depth_interval=2, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=32, pin_memory=True, num_workers=4)
model_dict = {'model': model, 'weight': 1, 'dataloader': loader_test}
models_dicts.append(model_dict)

model_name = 'medical_net_resnet50'
model_file = path_model / 'medical_net_resnet50.pth'
model = get_model(model_name, 1, model_file=model_file)
ds_test = Dataset_CSV_test(csv_file=csv_file, image_shape=args.image_shape,
                           depth_start=0, depth_interval=2, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=32, pin_memory=True, num_workers=4)
model_dict = {'model': model, 'weight': 1, 'dataloader': loader_test}
models_dicts.append(model_dict)


list_probs, probs_ensembling = predict_multiple_models(models_dicts)

path_pkl = Path(args.dir_dest) / 'probs_ensemble.pkl'
path_pkl.parent.mkdir(parents=True, exist_ok=True)
with path_pkl.open(mode='wb') as f:
    pickle.dump((list_probs, probs_ensembling), f)

labels_pd = np.array(probs_ensembling)
labels_pd[labels_pd > threshold] = 1
labels_pd[labels_pd <= threshold] = 0

df = pd.read_csv(csv_file)
(image_files, labels) = list(df['images']), list(df['labels'])
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels, labels_pd)
print(cf)

if args.export_confusion_files:
    from libs.neural_networks.helper.my_export_confusion_files import export_confusion_files_binary_class
    export_confusion_files_binary_class(image_files, labels, probs_ensembling, dir_original, dir_preprocess,
                                        args.dir_dest, threshold)
    print('export confusion files ok!')
