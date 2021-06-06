import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from libs.dataset.my_dataset_torchio import Dataset_CSV_test
from torch.utils.data import DataLoader
import pandas as pd

save_features = False

dir_dest = '/disk1/3D_OCT_AMD/2021_4_22/results/heatmaps/t_SNE/'

csv_file = os.path.join(os.path.abspath('../..'),
                'datafiles', 'v1_128_128_128', '3D_OCT_AMD_split_patid_test.csv')
tsne_image_file = os.path.join(dir_dest, 't_sne_test.png')


#region laod model
from libs.neural_networks.model.ModelsGenesis.unet3d import UNet3D_classification
num_class = 2
model = UNet3D_classification(n_class=num_class)
model_file = '/tmp2/2020_4_29/v1_128_128_128/ModelsGenesis/0/epoch11.pth'
state_dict = torch.load(model_file, map_location='cpu')
model.load_state_dict(state_dict, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.eval()

layer_features = model.dense_1
image_shape = (64, 64)
batch_size = 32

#endregion


ds_test = Dataset_CSV_test(csv_file=csv_file, image_shape=image_shape,
                           depth_start=0, depth_interval=2, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=batch_size,
                         pin_memory=True, num_workers=4)

from libs.neural_networks.heatmaps.t_SNE.my_tsne_helper import compute_features_files, gen_tse_features, draw_tsne
features = compute_features_files(model, layer_features, loader_test)

X_tsne = gen_tse_features(features)
if save_features:
    npy_file_features = "/disk1/share_8tb/广角眼底2021.04.12/results/T-SNE/test"
    os.makedirs(os.path.dirname(npy_file_features), exist_ok=True)
    import numpy as np
    np.save(npy_file_features, X_tsne)
    # X_tsne = np.load(save_npy_file)


draw_tsne(X_tsne, pd.read_csv(csv_file)['labels'], nb_classes=2, save_tsne_image=tsne_image_file,
             labels_text=['Normal', 'High Risk AMD'], colors=['g', 'r'])



print('OK')

