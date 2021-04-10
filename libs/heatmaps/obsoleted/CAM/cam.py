'''based on https://snappishproductions.com/blog/2018/01/03/class-activation-mapping-in-pytorch.html.html'''

import numpy as np
import torch
from torch.nn import functional as F
from torch import topk

class SaveFeatures():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.features = ((output.cpu()).gradients).numpy()

    def remove(self):
        self.hook.remove()

def __get_features(model, inputs, layer1):

    activated_features = SaveFeatures(layer1)

    prediction = model(inputs)
    pred_probabilities = F.softmax(prediction).data.squeeze()

    activated_features.remove()

    return activated_features.features, pred_probabilities

def __getCAM(feature_conv, weight_fc, class_idx, ndim):
    feature_conv = np.max(feature_conv, 0, keepdims=True) #some neural networks do not need this

    if ndim == 2:
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]
    if ndim == 3:
        _, nc, d, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, d*h*w)))
        cam = cam.reshape(d, h, w)
        cam = cam - np.min(cam)
        cam_3d = cam / np.max(cam)
        return [cam_3d]

def get_cam(model, inputs, layer_conv, layer_fc='fc', class_idx=None, ndim=2):
    #support both layer name and the reference of the layer.
    if isinstance(layer_conv, str):
        layer_conv = model._modules.get(layer_conv)
    if isinstance(layer_fc, str):
        layer_fc = model._modules.get(layer_fc)

    features, pred_probabilities = __get_features(model, inputs, layer_conv)

    weight_softmax_params = list(layer_fc.parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().gradients.numpy())
    if class_idx is None:
        class_idx = topk(pred_probabilities, 1)[1].int()

    overlay = __getCAM(features, weight_softmax, class_idx, ndim=ndim)
    overlay = np.squeeze(overlay)

    return overlay