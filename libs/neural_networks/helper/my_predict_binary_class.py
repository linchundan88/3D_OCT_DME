import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


def predict_single_model(model, dataloader, log_interval=1, activation='sigmoid', threshold=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    list_probs = []
    with torch.no_grad():
        for batch_idx, (inputs) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            if activation == 'sigmoid':
                outputs = F.sigmoid(outputs).data
            outputs = torch.flatten(outputs)
            list_probs.extend(outputs.cpu().numpy().tolist())

            if (log_interval is not None) and (batch_idx % log_interval == log_interval - 1):
                    print(f'batch:{batch_idx}')

    if threshold is not None:
        probs = np.array(list_probs)
        probs[probs > 0.5] = 1
        probs[probs <= 0.5] = 0
        list_labels = probs
        return list_probs, list_labels
    else:
        return list_probs



def predict_multiple_models(model_dicts, log_interval=1, activation='softmax'):
    list_probs = []
    total_weights = 0

    for model_dict in model_dicts:
        model = model_dict['model']
        dataloader = model_dict['dataloader']
        weight = model_dict['weight']

        probs = predict_single_model(model, dataloader, log_interval=log_interval, activation=activation, argmax=False)
        list_probs.append(probs)
        total_weights += weight
        if 'final_probs' not in locals().keys():
            final_probs = probs * weight
        else:
            final_probs += probs * weight

    probs_ensembling = final_probs / total_weights

    return list_probs, probs_ensembling


