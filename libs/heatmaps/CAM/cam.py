'''based on https://snappishproductions.com/blog/2018/01/03/class-activation-mapping-in-pytorch.html.html'''

import numpy as np
import torch
from torch.nn import functional as F
from torch import topk

class SaveFeatures():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()

def __get_features(model, inputs, layer_name):
    final_layer = model._modules.get(layer_name)
    # if len(model._modules.get(layer_name)._modules) > 0:  #sequential
    #     for k, v in arch._modules.get(layer_name)._modules.items():
    #         final_layer = v

    activated_features = SaveFeatures(final_layer)

    prediction = model(inputs)
    pred_probabilities = F.softmax(prediction).data.squeeze()

    activated_features.remove()

    return activated_features.features, pred_probabilities

def __getCAM(feature_conv, weight_fc, class_idx, dim=2):
    feature_conv = np.max(feature_conv, 0, keepdims=True) #some neural networks do not need this

    if dim == 2:
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]
    if dim == 3:
        _, nc, d, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, d*h*w)))
        cam = cam.reshape(d, h, w)
        cam = cam - np.min(cam)
        cam_3d = cam / np.max(cam)
        return [cam_3d]

def get_cam(model, inputs, layer_name_conv, layer_name_fc='fc', class_idx=None, dim=2):
    model.eval()
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")

    inputs = inputs.to(device)
    features, pred_probabilities = __get_features(model, inputs, layer_name_conv)

    weight_softmax_params = list(model._modules.get(layer_name_fc).parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    if class_idx is None:
        class_idx = topk(pred_probabilities, 1)[1].int()

    overlay = __getCAM(features, weight_softmax, class_idx, dim=dim)
    overlay = np.squeeze(overlay)

    return overlay