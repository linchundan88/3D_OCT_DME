import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import numpy as np
import torch.nn as nn
from captum.attr import *
import pandas as pd
import cv2
import shutil
from matplotlib import pyplot as plt


dir_original = '/disk1/3D_OCT_DME/original/'
dir_preprocess = '/disk1/3D_OCT_DME/preprocess/128_128_128/'
dir_dest = '/disk1/3D_OCT_DME/results/heatmaps/'

csv_file = os.path.join(os.path.abspath('../..'),
                'datafiles', 'v1_topocon_128_128_128', f'3D_OCT_DME_split_patid_test.csv')


#region laod model
image_shape = (64, 64)
from libs.neural_networks.ModelsGenesis.unet3d import UNet3D, TargetNet
base_model = UNet3D()
model = TargetNet(base_model, n_class=2)
# model_file = '/tmp2/2020_2_23/v1_topocon_128_128_128/ModelsGenesis/0/epoch12.pth'
model_file = '/tmp2/2020_2_23_afternoon/v1_topocon_128_128_128/medical_net_resnet50/0/epoch5.pth'
state_dict = torch.load(model_file, map_location='cpu')
model.load_state_dict(state_dict, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.eval()
#endregion


heatmap_type = 'GuidedBackprop'
if heatmap_type == 'GuidedBackprop':
    guidedBackprop = GuidedBackprop(model)
if heatmap_type == 'IntegratedGradients':
    integratedGradients = IntegratedGradients(model)


from libs.dataset.my_dataset_torchio import get_tensor
df = pd.read_csv(csv_file)
for index, row in df.iterrows():
    file_npy = row['images']
    tensor_x = get_tensor(file_npy, image_shape=image_shape,
                          depth_start=0, depth_interval=2)
    tensor_x = tensor_x.to(device)
    with torch.no_grad():
        outputs = model(tensor_x)
        outputs = torch.softmax(outputs, dim=1)
        #or pytorch implementation: _, preds = torch.max(outputs, 1)
        probs = outputs.cpu().numpy()
        class_predict = probs.argmax(axis=-1)[0] #array(size:1) -> int value

        if class_predict == 1:
            if heatmap_type == 'GuidedBackprop':
                attribution = guidedBackprop.attribute(tensor_x, target=1)
                # (B,C,D,H,W) linear(3D-only), bilinear(4D-only), trilinear(5D-only).
                attribution = nn.functional.interpolate(attribution, size=[128, 128, 128],
                                                        mode='trilinear')

                gradients = attribution.cpu().numpy()
                gradients = np.squeeze(gradients)
                gradients = np.maximum(0, gradients)  # only positive gradients
                value_max = np.max(gradients)
                # gradients = gradients - gradients.min()
                gradients /= value_max
                heatmaps = (gradients * 255).astype(np.uint8)

            # '/disk1/3D_OCT_DME/preprocess/128_128_128/Topocon/M0/02-000399_20161201_094159_OPT_L_001/02-000399_20161201_094159_OPT_L_001.npy'
            dirname, filename = os.path.split(file_npy)
            dir_dest_files = os.path.join(dir_dest,  dirname.replace(dir_preprocess, ''), 'images')
            array_3d = np.load(file_npy)  # shape (D,H,W)
            for i in range(array_3d.shape[0]):
                file_img = os.path.join(dir_dest_files, f'image_{str(i)}.jpg')
                os.makedirs(os.path.dirname(file_img), exist_ok=True)
                print(file_img)
                cv2.imwrite(file_img, array_3d[i])

            dir_dest_heatmaps = os.path.join(dir_dest,  dirname.replace(dir_preprocess, ''), 'heatmaps')
            for i in range(heatmaps.shape[0]):
                heatmap = heatmaps[i]
                file_heatmap = os.path.join(dir_dest_heatmaps, f'heatmap_{str(i)}.jpg')
                os.makedirs(os.path.dirname(file_heatmap), exist_ok=True)
                print(filename)
                cv2.imwrite(file_heatmap, heatmap)



print('OK')
