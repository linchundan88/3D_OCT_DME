'''a good atricle
https://gilberttanner.com/blog/interpreting-pytorch-models-with-captum
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import *

import torch
from torch import nn as nn
from torch.nn import functional as F
from libs.dataset.my_dataset_torchio import get_tensor

from libs.neural_networks.model.ModelsGenesis.unet3d import UNet3D, TargetNet
base_model = UNet3D()
model = TargetNet(base_model, n_class=2)
# model_file = '/tmp2/2020_2_23/v1_topocon_128_128_128/ModelsGenesis/0/epoch12.pth'
model_file = '/tmp2/2020_2_23_afternoon/v1_topocon_128_128_128/medical_net_resnet50/0/epoch5.pth'
state_dict = torch.load(model_file, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()


file_npy = '/disk1/3D_OCT_DME/preprocess/128_128_128/Topocon/Topocon/累及中央的黄斑水肿/02-m00407_20181206_115741_OPT_R_001/02-m00407_20181206_115741_OPT_R_001.npy'
inputs = get_tensor(file_npy, image_shape=(64, 64),
                    depth_start=0, depth_interval=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

inputs = inputs.to(device)
output = model(inputs)
probs = F.softmax(output, dim=1).data.cpu().numpy()
label_pd = probs.argmax(axis=-1)

print(label_pd)

label_pd = torch.IntTensor(label_pd)
label_pd = label_pd.to(device)

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

# IntegratedGradients Saliency  GuidedBackprop DeepLift, GradientShap,
# DeepLift, IntegratedGradients, LayerGradCam
heatmap_type = 'IntegratedGradients'

if heatmap_type == 'IntegratedGradients':
    integrated_gradients = IntegratedGradients(model)
    attribution = integrated_gradients.attribute(inputs, target=label_pd, n_steps=32)


print('OK')
