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
from libs.dataset.my_dataset_torchio import Dataset_CSV_test
from libs.neural_networks.helper.my_predict_multi_class import predict_single_model
from libs.neural_networks.model.my_get_model import get_model

filename_csv = Path(__file__).resolve().parent.parent.parent.joinpath('datafiles', 'v3', '3D_OCT_DME_M0_M1M2.csv')
dir_original = '/disk1/3D_OCT_DME/original/'
dir_preprocess = '/disk1/3D_OCT_DME/preprocess/128_128_128/'
dir_dest ='/disk1/3D_OCT_DME/results/confusion_files_M0_M1M2_2021_8_1/'
export_confusion_files = False

num_class = 2
model_name = 'cls_3d'
model_file = Path(__file__).resolve().parent.parent.joinpath(
    'trained_models', 'multi_class', 'cls_3d.pth')
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
    from libs import export_confusion_files_binary_class
    export_confusion_files_binary_class(image_files, labels_gt, probs, dir_original, dir_preprocess, dir_dest, threshold)
    print('export confusion files ok!')

