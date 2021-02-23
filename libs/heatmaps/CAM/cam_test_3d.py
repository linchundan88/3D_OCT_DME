
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn as nn
from torch.nn import functional as F
from libs.heatmaps.CAM.cam import get_cam
from libs.neural_networks.ModelsGenesis.unet3d import UNet3D_classification
from libs.dataset.my_dataset import npy_to_tensor
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet3D_classification(n_class=2)
model_file = '/tmp2/v1_topocon_64_64_64/ModelsGenesis/0/epoch6.pth'
state_dict = torch.load(model_file, map_location='cpu')
model.load_state_dict(state_dict, strict=False)

from libs.dataset.my_dataset import Dataset_CSV_3d
from torch.utils.data import DataLoader
filename_csv = os.path.join(os.path.abspath('../../..'),
                'datafiles', 'v1_topocon_64_64_64', f'3D_OCT_DME.csv')
batch_size_valid = 32
image_shape = (64, 64)
from libs.dataset.my_dataset import Dataset_CSV_3d_one
ds_valid = Dataset_CSV_3d(csv_file=filename_csv, image_shape=image_shape, test_mode=True)
loader_valid = DataLoader(ds_valid, batch_size=batch_size_valid,
                          num_workers=4)

# from libs.helper.my_predict import predict
# (probs, lables_pd) = predict(model, loader_valid)

'''
npy_file = '/disk1/3D_OCT_DME/preprocess/64_64_64/Topocon/Topocon/累及中央的黄斑水肿/02-001054_20180327_095457_OPT_L_001/02-001054_20180327_095457_OPT_L_001_d2_r0.npy'
image_shape = (64, 64)
tensor = npy_to_tensor(npy_file)
ds_valid = Dataset_CSV_3d_one(npy_file=npy_file, image_shape=image_shape, test_mode=True)
loader_valid = DataLoader(ds_valid, batch_size=1,
                          num_workers=1)

for batch_idx, (inputs) in enumerate(loader_valid):
    print(inputs.shape)
    dddd = 1
'''

model.train(False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.train(False)
model.eval()


ds_valid = Dataset_CSV_3d(csv_file=filename_csv, image_shape=image_shape, test_mode=True)
loader_valid = DataLoader(ds_valid, batch_size=32,
                          num_workers=1)
activation = 'softmax'
with torch.no_grad():
    for batch_idx, (inputs) in enumerate(loader_valid):
        inputs1 = inputs[0:4, :, :, :, :]
        inputs1 = inputs1.to(device)
        outputs1 = model(inputs1)
        outputs1 = F.softmax(outputs1, dim=1).data
        labels_pd1 = outputs1.argmax(axis=-1)
        print(labels_pd1)

        inputs1 = inputs[0:8, :, :, :, :]
        inputs1 = inputs1.to(device)
        outputs1 = model(inputs1)
        outputs1 = F.softmax(outputs1, dim=1).data
        labels_pd1 = outputs1.argmax(axis=-1)
        print(labels_pd1)

        inputs2 = inputs[0:16, :, :, :, :]
        inputs2 = inputs2.to(device)
        outputs2 = model(inputs2)
        outputs2 = F.softmax(outputs2, dim=1).data
        labels_pd2 = outputs2.argmax(axis=-1)
        print(labels_pd2)

        inputs3 = inputs[0:32, :, :, :, :]
        inputs3 = inputs3.to(device)
        outputs3 = model(inputs3)
        outputs3 = F.softmax(outputs3, dim=1).data
        labels_pd3 = outputs3.argmax(axis=-1)
        print(labels_pd3)

        for i in range(inputs.shape[0]):
            tensor1 = inputs[i]
            tensor2 = torch.unsqueeze(tensor1, dim=0)

            inputs1 = tensor2.to(device)
            outputs1 = model(inputs1)
            outputs1 = F.softmax(outputs1, dim=1).data
            print(outputs1)
            labels_pd1 = outputs1.argmax(axis=-1)
            print(labels_pd1)

        inputs = inputs.to(device)
        outputs = model(inputs)
        if activation == 'softmax':
            outputs = F.softmax(outputs, dim=1).data

        labels_pd = outputs.argmax(axis=-1)
        print(labels_pd.cpu().numpy())




print('OK')