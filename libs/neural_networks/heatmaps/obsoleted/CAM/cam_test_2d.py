'''https://snappishproductions.com/blog/2018/01/03/class-activation-mapping-in-pytorch.html.html'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
from matplotlib.pyplot import imshow, show
from torchvision import transforms
import skimage.transform
from libs.neural_networks.heatmaps.obsoleted import get_cam
import torch
from torch import nn as nn


img_path = os.path.join(os.path.abspath(''), 'cat.jpeg')
image = Image.open(img_path)
imshow(image)

# Imagenet mean/std

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

# Preprocessing - scale to 224x224 for model, convert to tensor,
# and normalize to -1..1 with mean/std for ImageNet

image_shape = (299, 299)

preprocess = transforms.Compose([
   transforms.Resize(image_shape),
   transforms.ToTensor(),
   normalize
])

display_transform = transforms.Compose([
   transforms.Resize(image_shape)])
tensor = preprocess(image)
tensor = tensor.unsqueeze(0)

# model = models.resnet18(pretrained=True)
# overlay = get_cam(model, inputs=tensor,
#                   layer_name_conv='layer4', layer_name_fc='fc')

import pretrainedmodels
#https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionresnetv2.py
# 'xception', 'inceptionresnetv2', 'inceptionv3'
# model_name = 'xception'
# model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# overlay = get_cam(model, inputs=tensor,
#                   layer_name_conv='conv4', layer_name_fc='last_linear')

# model_name = 'inceptionv3'
# model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# overlay = get_cam(model, inputs=tensor,
#                   layer_name_conv='Mixed_7c', layer_name_fc='last_linear')

model_name = 'inceptionresnetv2'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
   model.to(device)
if torch.cuda.device_count() > 1:
   model = nn.DataParallel(model)
tensor = tensor.to(device)

# overlay = get_cam(model, inputs=tensor,
#                   layer_conv='conv2d_7b', layer_name_fc='last_linear')

overlay = get_cam(model, inputs=tensor,
                  layer_conv=model.conv2d_7b, layer_fc='last_linear')


imshow(overlay, alpha=0.5, cmap='jet')
show()
imshow(display_transform(image))
tensor = tensor.squeeze()
imshow(skimage.transform.resize(overlay, tensor.shape[1:3]), alpha=0.5, cmap='jet');

show()

print('OK')