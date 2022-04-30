import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

@torch.no_grad()
def predict_single_model(model, dataloader, log_interval=1, activation='softmax', argmax=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    list_probs = []
    for batch_idx, (inputs) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        if activation == 'softmax':
            outputs = F.softmax(outputs, dim=1).data
        list_probs.append(outputs.cpu().numpy())

        if (log_interval is not None) and (batch_idx % log_interval == log_interval - 1):
                print(f'batch:{batch_idx}')

    probs = np.vstack(list_probs)
    if argmax:
        labels = probs.argmax(axis=-1)
        return probs, labels
    else:
        return probs


def predict_multiple_models(model_dicts, log_interval=1, activation='softmax'):
    list_probs = []
    total_weights = 0

    for model_dict in model_dicts:
        model = model_dict['model']
        dataloader = model_dict['dataloader']
        weight = model_dict['weight']
        total_weights += weight

        probs = predict_single_model(model, dataloader, log_interval=log_interval, activation=activation, argmax=False)
        list_probs.append(probs)
        if 'final_probs' not in locals().keys():
            final_probs = probs * weight
        else:
            final_probs += probs * weight

    probs_ensembling = final_probs / total_weights

    return list_probs, probs_ensembling


