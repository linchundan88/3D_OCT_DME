import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class _SaveFeatures():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()


@torch.no_grad()
def __get_features(model, inputs, layer):
    if isinstance(layer, str):
        layer = model._modules.get(layer)

    # if len(model._modules.get(layer_name)._modules) > 0:  #sequential
    #     for k, v in arch._modules.get(layer_name)._modules.items():
    #         final_layer = v

    activated_features = _SaveFeatures(layer)
    _ = model(inputs)
    activated_features.remove()

    return activated_features.features

@torch.no_grad()
def compute_features(model, inputs, layer):
    model.eval()
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")
    inputs = inputs.to(device)
    features = __get_features(model, inputs, layer)

    return features

@torch.no_grad()
def compute_features_files(model, layer, data_loader):
    model.eval()
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")

    for batch_idx, inputs in enumerate(data_loader):
        print('batch:', batch_idx)

        inputs = inputs.to(device)
        features = __get_features(model, inputs, layer)

        if 'features_all' not in locals().keys():
            features_all = features
        else:
            features_all = np.concatenate((features_all, features), axis=0)

    return features_all


# colors: b--blue, c--cyan, g--green, k--black, r--red, w--white, y--yellow, m--magenta
def draw_tsne(X_tsne, labels, nb_classes, labels_text, colors=['g', 'r'], save_tsne_image=None):

    y = np.array(labels)
    colors_map = y

    plt.figure(figsize=(10, 10))
    for cl in range(nb_classes):
        indices = np.where(colors_map == cl)
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=colors[cl], label=labels_text[cl])
    plt.legend()

    os.makedirs(os.path.dirname(save_tsne_image), exist_ok=True)
    plt.savefig(save_tsne_image)
    # plt.show()


def gen_tse_features(features):
    tsne = TSNE(n_components=2)  #n_components:Dimension of the embedded space.
    X_tsne = tsne.fit_transform(features)

    return X_tsne