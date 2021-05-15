import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import torch.nn as nn
from captum.attr import *
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import imageio


dir_original = '/disk1/3D_OCT_DME/original/128_128_128/'
dir_preprocess = '/disk1/3D_OCT_DME/preprocess/128_128_128/'
dir_dest = '/disk1/3D_OCT_DME/results/heatmaps/test/'

csv_file = os.path.join(os.path.abspath('../..'),
                'datafiles', 'v1_topocon_128_128_128', '3D_OCT_DME_split_patid_test.csv')

upsample_size = (128, 128, 128)
gif_fps = 1

image_shape = (64, 64)

#region laod model

from libs.neural_networks.model.ModelsGenesis.cls3d import Cls_3d
num_class = 2
model = Cls_3d(n_class=num_class)
model_file = '/tmp2/2020_5_15/v1_topocon_128_128_128/ModelsGenesis/0/epoch12.pth'
state_dict = torch.load(model_file, map_location='cpu')
model.load_state_dict(state_dict, strict=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    model.to(device)
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
model = model.eval()
#endregion


heatmap_type = 'GuidedGradCam'
noise_tunnel = False

dir_dest = os.path.join(dir_dest, heatmap_type)
if heatmap_type == 'GuidedBackprop':
    guidedBackprop = GuidedBackprop(model)
if heatmap_type == 'IntegratedGradients':
    integratedGradients = IntegratedGradients(model)
if heatmap_type == 'LayerGradCam':
    layer_gc = LayerGradCam(model, model.down_tr512)
if heatmap_type == 'GuidedGradCam':
    guidedGradCam = GuidedGradCam(model, model.down_tr512)

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
        class_predict = int(probs.argmax(axis=-1)[0]) #array(size:1) -> int value

        if class_predict == 1:
            array_3d = np.load(file_npy)  # shape (D,H,W), used to generate original images

            if heatmap_type in ['GuidedBackprop', 'IntegratedGradients', 'GuidedGradCam']:

                if heatmap_type == 'GuidedBackprop':
                    if not noise_tunnel:
                        attribution = guidedBackprop.attribute(tensor_x, target=class_predict)
                    else: #in this study, using NoiseTunel generate bad results.
                        nt = NoiseTunnel(guidedBackprop)
                        attribution = nt.attribute(tensor_x, nt_type='smoothgrad',
                                                   stdevs=0.05, nt_samples=20, target=class_predict)

                if heatmap_type == 'IntegratedGradients':
                    attribution = integratedGradients.attribute(tensor_x, target=class_predict)
                    # nt = NoiseTunnel(integratedGradients)
                    # attribution = nt.attribute(tensor_x, nt_type='smoothgrad',
                    #                            nt_samples=1, target=class_predict)

                if heatmap_type == 'GuidedGradCam':
                    attribution = guidedGradCam.attribute(tensor_x, target=class_predict)


                # (B,C,D,H,W) linear(3D-only), bilinear(4D-only), trilinear(5D-only).
                attribution = nn.functional.interpolate(attribution, size=upsample_size,
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
                dir_heatmaps = os.path.join(dir_dest, dirname.replace(dir_preprocess, ''))
                for i in range(array_3d.shape[0]):
                    #original image
                    file_img = os.path.join(dir_heatmaps, f'image_{str(i)}.jpg')
                    os.makedirs(os.path.dirname(file_img), exist_ok=True)
                    print(file_img)
                    cv2.imwrite(file_img, array_3d[i])

                    #heatmap image
                    heatmap = heatmaps[i]
                    file_heatmap = os.path.join(dir_heatmaps, f'heatmap_{str(i)}.jpg')
                    os.makedirs(os.path.dirname(file_heatmap), exist_ok=True)
                    print(filename)
                    cv2.imwrite(file_heatmap, heatmap)

                    # heatmaps gif
                    file_img = os.path.join(dir_heatmaps, f'image_{str(i)}.jpg')
                    file_heatmap = os.path.join(dir_heatmaps, f'heatmap_{str(i)}.jpg')

                    mg_paths = [file_img, file_heatmap]
                    gif_images = []
                    for path in mg_paths:
                        gif_images.append(imageio.imread(path))
                    file_heatmap_gif = os.path.join(dir_heatmaps, f'heatmap_gif_{str(i)}.gif')
                    os.makedirs(os.path.dirname(file_heatmap_gif), exist_ok=True)
                    imageio.mimsave(file_heatmap_gif, gif_images, fps=gif_fps)


            if heatmap_type == 'LayerGradCam':
                attr = layer_gc.attribute(tensor_x, class_predict, relu_attributions=True)
                upsampled_attr = LayerAttribution.interpolate(attr, upsample_size, interpolate_mode='trilinear')
                cam = upsampled_attr.detach().cpu().numpy().squeeze()

                cam = np.maximum(cam, 0)  # using relu_attributions=True
                cam = cam / np.max(cam)  # heatmap:0-1

                dirname, filename = os.path.split(file_npy)
                dir_heatmaps = os.path.join(dir_dest, dirname.replace(dir_preprocess, ''))

                #region save heatmaps
                for i in range(array_3d.shape[0]):
                    #original image
                    file_img = os.path.join(dir_heatmaps, f'image_{str(i)}.jpg')
                    os.makedirs(os.path.dirname(file_img), exist_ok=True)
                    print(file_img)
                    cv2.imwrite(file_img, array_3d[i])

                    #heatmap image
                    file_heatmap = os.path.join(dir_heatmaps, f'heatmap_{str(i)}.jpg')
                    os.makedirs(os.path.dirname(file_heatmap), exist_ok=True)
                    print(filename)

                    plt.axis("off")  # turns off axes
                    # plt.axis("tight")  # gets rid of white border
                    plt.imshow(cam[i], alpha=0.5, cmap='jet')
                    # plt.show()
                    #cam = cv2.applyColorMap(np.uint8(255 * grads), cv2.COLORMAP_JET)
                    plt.savefig(file_heatmap, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    # heatmaps gif
                    file_img = os.path.join(dir_heatmaps, f'image_{str(i)}.jpg')
                    file_heatmap = os.path.join(dir_heatmaps, f'heatmap_{str(i)}.jpg')

                    mg_paths = [file_img, file_heatmap]
                    gif_images = []
                    for path in mg_paths:
                        gif_images.append(imageio.imread(path))
                    file_heatmap_gif = os.path.join(dir_heatmaps, f'heatmap_gif_{str(i)}.gif')
                    os.makedirs(os.path.dirname(file_heatmap_gif), exist_ok=True)
                    imageio.mimsave(file_heatmap_gif, gif_images, fps=gif_fps)


print('OK')
