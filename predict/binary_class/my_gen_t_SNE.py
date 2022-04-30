
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import pandas as pd
from libs.dataset.my_dataset_torchio import Dataset_CSV_test
from torch.utils.data import DataLoader
from libs.neural_networks.model.my_get_model import get_model
from libs.neural_networks.heatmaps.t_SNE.my_tsne_helper import compute_features_files, gen_tse_features, draw_tsne
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task_type', default='3D_OCT_DME_M0_M1M2_test') #3D_OCT_DME_M0_M1M2, 3D_OCT_DME_M1_M2
parser.add_argument('--data_version', default='v1')
parser.add_argument('--model_name', default='cls_3d')
parser.add_argument('--image_shape', default=(64, 64))
parser.add_argument('--dir_dest', default='/tmp2/3D_OCT_DME/2022_4_28_64_64/m0_m1m2')

args = parser.parse_args()


#region load model and set some parameters
csv_file = Path(__file__).resolve().parent.parent.parent / 'datafiles' / args.data_version / f'{args.task_type}.csv'
tsne_image_file = os.path.join(args.dir_dest, f't_sne_{args.model_name}.png')
save_features = False
npy_file_features = os.path.join(args.dir_dest, 't_sne_test.npy')

path_model_base = Path(__file__).resolve().parent.parent.parent / 'trained_models' / '2022_4_28_64_64'
if args.task_type.startswith('3D_OCT_DME_M0_M1M2'):
    path_model = path_model_base / 'binary_class_m0_m1m2'
if args.task_type.startswith('3D_OCT_DME_M1_M2'):
    path_model = path_model_base / 'binary_class_m1_m2'

if args.model_name == 'cls_3d':
    model_name = 'cls_3d'
    model_file = path_model / 'cls_3d.pth'
if args.model_name == 'medical_net_resnet50':
    model_name = 'medical_net_resnet50'
    model_file = path_model / 'medical_net_resnet50.pth'

model = get_model(args.model_name, num_class=1, model_file=model_file)

layer_features = model.dense_1  #Cls_3d, medical_net_resnet50, dense_1
batch_size = 32
ds_test = Dataset_CSV_test(csv_file=csv_file, image_shape=args.image_shape,
                           depth_start=0, depth_interval=2, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=batch_size,
                         pin_memory=True, num_workers=4)
#endregion


features = compute_features_files(model, layer_features, loader_test)

X_tsne = gen_tse_features(features)
if save_features:
    os.makedirs(os.path.dirname(npy_file_features), exist_ok=True)
    import numpy as np
    np.save(npy_file_features, X_tsne)
    # X_tsne = np.load(save_npy_file)


draw_tsne(X_tsne, pd.read_csv(csv_file)['labels'], nb_classes=2, save_tsne_image=tsne_image_file,
             labels_text=['Normal', 'DME'], colors=['g', 'r'])



print('OK')

