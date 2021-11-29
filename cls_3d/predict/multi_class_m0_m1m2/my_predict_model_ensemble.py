import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent.parent)
from torch.utils.data import DataLoader
import pandas as pd
from cls_3d.libs.dataset.my_dataset_torchio import Dataset_CSV_test
from cls_3d.libs.neural_networks.helper.my_predict_multi_class import predict_multiple_models
from cls_3d.libs.neural_networks.model.my_get_model import get_model

filename_csv = Path(__file__).resolve().parent.parent.parent.joinpath('datafiles', 'v3', '3D_OCT_DME_M0_M1M2.csv')
dir_original = '/disk1/3D_OCT_DME/original/'
dir_preprocess = '/disk1/3D_OCT_DME/preprocess/128_128_128/'
dir_dest = '/tmp2/3D_OCT_DME/3D_OCT_DME_confusion_files_2021_6_8/'
export_confusion_files = False

num_class = 2

models_dicts = []
model_name = 'cls_3d'
model_file = Path(__file__).resolve().parent.parent.parent.joinpath(
    'trained_models', 'multi_class_m0_m1m2', 'cls_3d.pth')
model = get_model(model_name, num_class, model_file=model_file)
image_shape = (64, 64)
ds_test = Dataset_CSV_test(csv_file=filename_csv, image_shape=image_shape,
                           depth_start=0, depth_interval=2, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=32, pin_memory=True, num_workers=4)
model_dict = {'model': model, 'weight': 1, 'dataloader': loader_test}
models_dicts.append(model_dict)

model_name = 'medical_net_resnet50'
# model_file = '/tmp2/2021_6_6/v2/medical_net_resnet50/0/epoch13.pth'
model_file = Path(__file__).parent.parent.parent.joinpath(
    'trained_models', 'multi_class_m0_m1m2', 'medical_net_resnet50.pth')
model = get_model(model_name, num_class, model_file=model_file)
image_shape = (64, 64)
ds_test = Dataset_CSV_test(csv_file=filename_csv, image_shape=image_shape,
                           depth_start=0, depth_interval=2, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=32, pin_memory=True, num_workers=4)
model_dict = {'model': model, 'weight': 1, 'dataloader': loader_test}
models_dicts.append(model_dict)


probs, probs_ensembling = predict_multiple_models(models_dicts, activation='softmax')
labels_pd = probs_ensembling.argmax(axis=-1)

df = pd.read_csv(filename_csv)
(image_files, labels) = list(df['images']), list(df['labels'])
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels, labels_pd)
print(cf)


from cls_3d.libs.neural_networks.helper.my_export_confusion_files import export_confusion_files_binary_class
export_confusion_files_binary_class(image_files, labels, probs_ensembling, dir_original, dir_preprocess, dir_dest, threshold)
print('export confusion files ok!')
