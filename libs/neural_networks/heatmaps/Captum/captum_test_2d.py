'''a good atricle
https://gilberttanner.com/blog/interpreting-pytorch-models-with-captum
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
from PIL import Image
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch.nn as nn
from torchvision import transforms

from captum.attr import *
from captum.attr import visualization as viz
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 'xception', 'inceptionresnetv2', 'inceptionv3'
model_name = 'inceptionresnetv2'
import pretrainedmodels
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.eval()

labels_path = 'imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)


# transform = transforms.Compose([
#  transforms.Resize(256),
#  transforms.CenterCrop(224),
#  transforms.ToTensor()
# ])

transform = transforms.Compose([
 transforms.Resize(320),
 transforms.CenterCrop(299),
 transforms.ToTensor()
])

transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

img_path = os.path.join(os.path.abspath(''), 'elephant.jpeg')
img = Image.open(img_path)

transformed_img = transform(img)

input = transform_normalize(transformed_img)
input = input.unsqueeze(0)

input = input.to(device)

output = model(input)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)

pred_label_idx.squeeze_()
predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

# dl = DeepLift(model)
# attribution_deeplift = dl.attribute(input, target=pred_label_idx)

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

# LayerGradCam
# IntegratedGradients Saliency  GuidedBackprop DeepLift, GradientShap,
# DeepLift, IntegratedGradients

heatmap_type = 'LayerGradCam'

if heatmap_type == 'IntegratedGradients':
    integrated_gradients = IntegratedGradients(model)
    #region code_test performance
    '''
    for i in range(10):
        print(i)
        print('start')
        print(time.time())
        attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)
        print(time.time())
        print('end')
    '''
    #endregion
    attribution = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=50)
    _ = viz.visualize_image_attr(np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 outlier_perc=1)
    exit(0)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    print(time.time())
    attributions_ig_nt = noise_tunnel.attribute(input, n_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
    print(time.time())
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          cmap=default_cmap,
                                          show_colorbar=True)

if heatmap_type == 'Saliency':
    saliency = Saliency(model)
    attribution = saliency.attribute(input, target=pred_label_idx)
    _ = viz.visualize_image_attr_multiple(np.transpose(attribution.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          cmap=default_cmap,
                                          show_colorbar=True)

if heatmap_type == 'GuidedBackprop':
    guidedBackprop = GuidedBackprop(model)
    attribution = guidedBackprop.attribute(input, target=pred_label_idx)
    _ = viz.visualize_image_attr_multiple(np.transpose(attribution.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          cmap=default_cmap,
                                          show_colorbar=True)

if heatmap_type == 'DeepLift':
    deepLift = DeepLift(model)
    # attribution = deepLift.attribute(input, baselines=input * 0, target=pred_label_idx)
    attribution = deepLift.attribute(input, target=pred_label_idx)
    attr_dl = np.transpose(attribution.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    _ = viz.visualize_image_attr_multiple(np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      cmap=default_cmap,
                                      show_colorbar=True)

if heatmap_type == 'DeepLiftShap':
    deepLiftShap = DeepLiftShap(model)
    baselines_input = torch.cat([input * 0, input * 0], dim=0)
    attribution = deepLiftShap.attribute(input,  baselines=baselines_input, target=pred_label_idx)
    _ = viz.visualize_image_attr_multiple(np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      cmap=default_cmap,
                                      show_colorbar=True)

if heatmap_type == 'LayerGradCam':
    # inceptionresnetv2 conv2d_7b, inceptionv3 Mixed_7c, xception conv4
    layerGradCam = LayerGradCam(model, model.conv2d_7b)
    attribution = layerGradCam.attribute(input, target=pred_label_idx)
    import torch.nn.functional as F
    attribution = F.relu(attribution)
    # (1,3,299,299) (1,1,10,10)
    attribution = F.interpolate(attribution, size=(299, 299), mode='bilinear', align_corners=False)
    saliency_map_min, saliency_map_max = attribution.min(), attribution.max()
    attribution = (attribution - saliency_map_min).div(saliency_map_max - saliency_map_min).gradients

    attribution = attribution.squeeze()
    attribution = attribution.cpu().numpy()
    import matplotlib.pyplot as plt

    img = plt.imshow(attribution, alpha=0.5, cmap='jet')
    from matplotlib.pyplot import imshow, show
    # plt.imsave("foo.png", attribution, format="png", cmap="jet")
    plt.axis("off")  # turns off axes
    plt.axis("tight")  # gets rid of white border
    plt.axis("image")  # square up the image instead of filling the "figure" space
    plt.savefig("test.png", bbox_inches='tight', pad_inches=0)
    show()



print('OK')
