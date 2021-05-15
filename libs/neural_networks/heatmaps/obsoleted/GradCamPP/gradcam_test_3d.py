import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn as nn
from torch.nn import functional as F
from libs.dataset.my_dataset_torchio import get_tensor
import matplotlib.pyplot as plt

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

output = model(inputs)
probs = F.softmax(output, dim=1).data.cpu().numpy()
label_pd = probs.argmax(axis=-1)

print(label_pd)

layer_conv = model.base_model.down_tr512.ops[1].conv1
from libs.neural_networks.heatmaps.obsoleted.GradCamPP import GradCAM
gradcam = GradCAM(model, layer_conv, dim=3)
mask, logit = gradcam(inputs, label_pd)
# mask, logit = gradcam(inputs, class_idx=label_pd)
# mask, logit = gradcam(normed_torch_img)  #class_idx N

mask = torch.squeeze(mask).cpu().numpy()

for i in range(mask.shape[0]):
    image = mask[i, :, :]
    from matplotlib.pyplot import imshow, show
    imshow(mask, alpha=0.5, cmap='jet')
    show()
    plt.axis("off")  # turns off axes
    plt.axis("tight")  # gets rid of white border
    plt.axis("image")  # square up the image instead of filling the "figure" space
    plt.savefig("test.png", bbox_inches='tight', pad_inches=0)

    print('aaa')

print('OK')