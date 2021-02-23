import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


def predict(model, dataloader, log_interval=1, activation='softmax', argmax=True):
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
            if activation == 'softmax':
                outputs = F.softmax(outputs, dim=1).data
            if activation == 'sigmoid':
                outputs = F.sigmoid(outputs).data
            list_probs.append(outputs.cpu().numpy())

            if log_interval is not None:
                if batch_idx % log_interval == log_interval - 1:
                    print(f'batch:{batch_idx}')

    probs = np.vstack(list_probs)

    if argmax:
        labels_pd = probs.argmax(axis=-1)
        return probs, labels_pd
    else:
        return probs