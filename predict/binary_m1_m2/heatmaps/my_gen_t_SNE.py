
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
from libs.dataset.my_dataset_torchio import Dataset_CSV_test
from torch.utils.data import DataLoader
from libs.neural_networks.model.my_get_model import get_model

#region load model and set some parameters
csv_file = os.path.join(os.path.abspath('../../..'), 'datafiles', 'v3', '3D_OCT_DME_M1_M2_test.csv')
dir_dest = '/disk1/3D_OCT_DME/results/2021_7_31/heatmaps_binary_class_m1_m2/t_SNE/'
tsne_image_file = os.path.join(dir_dest, 't_sne_test.png')
save_features = False
npy_file_features = os.path.join(dir_dest, 't_sne_test.npy')


model_name = 'cls_3d'
model_file = os.path.join(os.path.abspath('../../..'), 'trained_models', 'binary_class_m1_m2', 'cls_3d.pth')
image_shape = (64, 64)
model = get_model(model_name, num_class=1, model_file=model_file)
layer_features = model.dense_1  #Cls_3d, medical_net_resnet50, dense_1
batch_size = 32
ds_test = Dataset_CSV_test(csv_file=csv_file, image_shape=image_shape,
                           depth_start=0, depth_interval=2, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=batch_size,
                         pin_memory=True, num_workers=4)
#endregion

from libs.neural_networks.heatmaps.t_SNE.my_tsne_helper import compute_features_files, gen_tse_features, draw_tsne
features = compute_features_files(model, layer_features, loader_test)

X_tsne = gen_tse_features(features)
if save_features:
    os.makedirs(os.path.dirname(npy_file_features), exist_ok=True)
    import numpy as np
    np.save(npy_file_features, X_tsne)
    # X_tsne = np.load(save_npy_file)


draw_tsne(X_tsne, pd.read_csv(csv_file)['labels'], nb_classes=2, save_tsne_image=tsne_image_file,
             labels_text=['M1', 'M2'], colors=['g', 'r'])



print('OK')

