import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
from captum.attr import *
import imageio
from libs.neural_networks.model.my_get_model import get_model
from matplotlib import pyplot as plt
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task_type', default='3D_OCT_DME_M0_M1M2_train') #3D_OCT_DME_M0_M1M2, 3D_OCT_DME_M1_M2, train,valid,test
parser.add_argument('--data_version', default='v1')
parser.add_argument('--model_name', default='medical_net_resnet50')  #cls_3d, medical_net_resnet50
parser.add_argument('--heatmap_type', default='IntegratedGradients')  #GuidedBackprop GuidedGradCam IntegratedGradients
parser.add_argument('--image_shape', nargs='+', type=int, default=(64, 64))
parser.add_argument('--dir_dest', default='/tmp2/3D_OCT_DME/2022_4_28_64_64/m0_m1m2/medical_net_resnet50/IntegratedGradients')
args = parser.parse_args()

#region set parameters and load model
dir_original = '/disk1/3D_OCT_DME/original/'
dir_preprocess = '/disk1/3D_OCT_DME/preprocess_128_128_128/'
csv_file = Path(__file__).resolve().parent.parent.parent / 'datafiles' / args.data_version / f'{args.task_type}.csv'
upsample_size = (128, 128, 128)
slices_significance_number = 5
gif_fps = 1

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
model = get_model(model_name, 1, model_file=model_file)


class_target = 0  #binary classification, only one class
activation = 'sigmoid'
threshold = 0.5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.eval()
#endregion


noise_tunnel = False

if args.heatmap_type == 'GuidedBackprop':
    guidedBackprop = GuidedBackprop(model)
if args.heatmap_type == 'IntegratedGradients':
    integratedGradients = IntegratedGradients(model)
if args.heatmap_type == 'LayerGradCam':
    if model_name == 'cls_3d':
        layer_gc = LayerGradCam(model, model.down_tr512)
    if model_name == 'ModelsGenesis':
        layer_gc = LayerGradCam(model, model.base_model.out512)
    if model_name == 'medical_net_resnet50':
        layer_gc = LayerGradCam(model, model.base_model.layer4)
if args.heatmap_type == 'GuidedGradCam':
    if model_name == 'cls_3d':
        guidedGradCam = GuidedGradCam(model, model.down_tr512)
    if model_name == 'ModelsGenesis':
        guidedGradCam = GuidedGradCam(model, model.base_model.out512)
    if model_name == 'medical_net_resnet50':
        guidedGradCam = GuidedGradCam(model, model.base_model.layer4)



from libs.dataset.my_dataset_torchio import get_tensor
df = pd.read_csv(csv_file)
for index, row in df.iterrows():
    file_npy, class_gt = row['images'], row['labels']
    tensor_x = get_tensor(file_npy, image_shape=args.image_shape,
                          depth_start=0, depth_interval=2)
    tensor_x = tensor_x.to(device)
    with torch.no_grad():
        outputs = model(tensor_x)
        if activation == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        prob = float(outputs)

        if prob > threshold:
            array_3d = np.load(file_npy)  # shape (D,H,W), used to generate the preprocessed images

            if args.heatmap_type in ['GuidedBackprop', 'IntegratedGradients', 'GuidedGradCam']:

                if args.heatmap_type == 'GuidedBackprop':
                    if not noise_tunnel:
                        attribution = guidedBackprop.attribute(tensor_x, target=class_target)
                    else: #in this study, using NoiseTunel generate bad results.
                        nt = NoiseTunnel(guidedBackprop)
                        attribution = nt.attribute(tensor_x, nt_type='smoothgrad',
                                                   stdevs=0.05, nt_samples=20, target=class_target)

                if args.heatmap_type == 'IntegratedGradients':
                    if not noise_tunnel:
                        attribution = integratedGradients.attribute(tensor_x, target=class_target)
                    else:
                        nt = NoiseTunnel(integratedGradients)
                        attribution = nt.attribute(tensor_x, nt_type='smoothgrad',
                                                   stdevs=0.05, nt_samples=4, target=class_target)

                if args.heatmap_type == 'GuidedGradCam':
                    attribution = guidedGradCam.attribute(tensor_x, target=class_target)


                # (B,C,D,H,W) linear(3D-only), bilinear(4D-only), trilinear(5D-only).
                attribution = nn.functional.interpolate(attribution, size=upsample_size,
                                                        mode='trilinear')

                gradients = attribution.cpu().numpy()
                gradients = np.squeeze(gradients, axis=(0,1))  #(N,C,D,H,W) - >(D,H,W)
                gradients = np.maximum(0, gradients)  # only positive gradients
                value_max = np.max(gradients)
                # gradients = gradients - gradients.min()
                gradients /= value_max
                heatmaps = (gradients * 255).astype(np.uint8)

                # find the most 5 significant heatmap slices
                avg1 = np.average(heatmaps, axis=(1,2))
                slice_significance = np.argsort(-avg1)[:slices_significance_number]
                # slice_significance2 = np.argpartition(avg1, -5)[-5:]
                # import heapq
                # slice_significance = heapq.nlargest(slices_significance_number, range(len(avg1)), avg1.take)

                # '/disk1/3D_OCT_DME/preprocess/128_128_128/Topocon/M0/02-000399_20161201_094159_OPT_L_001/02-000399_20161201_094159_OPT_L_001.npy'
                dirname, filename = os.path.split(file_npy)
                dir_heatmaps = os.path.join(args.dir_dest, dirname.replace(dir_preprocess, ''))
                os.makedirs(os.path.dirname(dir_heatmaps), exist_ok=True)
                for i in range(array_3d.shape[0]):
                    #file_preprocessed image
                    file_preprocessed = os.path.join(dir_heatmaps, f'image_{str(i)}.jpg')
                    os.makedirs(os.path.dirname(file_preprocessed), exist_ok=True)
                    print(file_preprocessed)
                    cv2.imwrite(file_preprocessed, array_3d[i])

                    if i in slice_significance:
                        # heatmap image
                        file_heatmap = os.path.join(dir_heatmaps, f'heatmap_{str(i)}.jpg')
                        print(file_heatmap)
                        cv2.imwrite(file_heatmap, heatmaps[i])

                        # heatmaps gif
                        mg_paths = [file_preprocessed, file_heatmap]
                        gif_images = []
                        for path in mg_paths:
                            gif_images.append(imageio.imread(path))
                        file_heatmap_gif = os.path.join(dir_heatmaps, f'heatmap_gif_{str(i)}.gif')
                        print(file_heatmap_gif)
                        imageio.mimsave(file_heatmap_gif, gif_images, fps=gif_fps)


            if args.heatmap_type == 'LayerGradCam':
                attr = layer_gc.attribute(tensor_x, class_target, relu_attributions=True)
                upsampled_attr = LayerAttribution.interpolate(attr, upsample_size, interpolate_mode='trilinear')
                cam = upsampled_attr.detach().cpu().numpy().squeeze()

                cam = np.maximum(cam, 0)  # using relu_attributions=True
                cam = cam / np.max(cam)  # heatmap:0-1

                dirname, filename = os.path.split(file_npy)
                dir_heatmaps = os.path.join(args.dir_dest, dirname.replace(dir_preprocess, ''))

                #region save heatmaps
                for i in range(array_3d.shape[0]):
                    #original image
                    file_img = os.path.join(dir_heatmaps, f'image_{str(i)}.jpg')
                    os.makedirs(os.path.dirname(file_img), exist_ok=True)
                    print(file_img)
                    cv2.imwrite(file_img, array_3d[i])

                    if i in slice_significance:
                        #heatmap image
                        file_heatmap = os.path.join(dir_heatmaps, f'heatmap_{str(i)}.jpg')
                        os.makedirs(os.path.dirname(file_heatmap), exist_ok=True)

                        plt.axis("off")  # turns off axes
                        # plt.axis("tight")  # gets rid of white border
                        plt.imshow(cam[i], alpha=0.5, cmap='jet')
                        # plt.show()
                        #cam = cv2.applyColorMap(np.uint8(255 * grads), cv2.COLORMAP_JET)
                        print(file_heatmap)
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
                        print(file_heatmap_gif)
                        imageio.mimsave(file_heatmap_gif, gif_images, fps=gif_fps)


print('OK')
