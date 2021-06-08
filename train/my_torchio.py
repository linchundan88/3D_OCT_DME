import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
from imgaug import augmenters as iaa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from libs.dataset.my_dataset_torchio import Dataset_CSV_train, Dataset_CSV_test
from torch.utils.data import DataLoader
from libs.neural_networks.helper.my_train import train
from libs.neural_networks.model.my_get_model import get_model


#region prepare dataset
task_type = '3D_OCT_DME'
image_shape = (64, 64)
data_version = 'v1_topocon_128_128_128'
data_version = 'v2'
csv_train = os.path.join(os.path.abspath('..'),
                'datafiles', data_version, f'{task_type}_train.csv')
csv_valid = os.path.join(os.path.abspath('..'),
                'datafiles', data_version, f'{task_type}_valid.csv')
csv_test = os.path.join(os.path.abspath('..'),
                'datafiles', data_version, f'{task_type}_test.csv'.format(data_version))

'''


'''

imgaug_iaa = iaa.Sequential([
    # iaa.Fliplr(0.5),

    # iaa.GaussianBlur(sigma=(0.0, 0.3)),
    # iaa.MultiplyBrightness(mul=(0.8, 1.2)),
    # iaa.contrast.LinearContrast((0.8, 1.2)),

    iaa.Sometimes(0.9, iaa.Add((-8, 8))),
    # iaa.Sometimes(0.9, iaa.Affine(
    #     scale=(0.98, 1.02),
    #     translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
    #     rotate=(-5, 5),
    # )),
])

df = pd.read_csv(csv_train)
num_class = df['labels'].nunique(dropna=True)

from torch.utils.data.sampler import WeightedRandomSampler
# list_class_samples = []
# for label in range(num_class):
#     list_class_samples.append(len(df[df['labels'] == label]))
# sampling_class_weights = 1 / np.power(list_class_samples, 0.5)
sampling_weights = []
sampling_class_weights = [1, 2]
for label in df['labels']:
    sampling_weights.append([1, 2][label])
sampler = WeightedRandomSampler(weights=sampling_weights, num_samples=len(df))

batch_size_train, batch_size_valid = 32, 32
num_workers = 4  #recommend num_workers = the number of gpus * 4

ds_train = Dataset_CSV_train(csv_file=csv_train, image_shape=image_shape)
loader_train = DataLoader(ds_train, batch_size=batch_size_train,
                          sampler=sampler, pin_memory=True, num_workers=num_workers)
ds_valid = Dataset_CSV_test(csv_file=csv_valid, image_shape=image_shape,
                            depth_start=0, depth_interval=2)
loader_valid = DataLoader(ds_valid, batch_size=batch_size_valid,
                           pin_memory=True, num_workers=num_workers)
ds_test = Dataset_CSV_test(csv_file=csv_test, image_shape=image_shape,
                           depth_start=0, depth_interval=2)
loader_test = DataLoader(ds_test, batch_size=batch_size_valid,
                            pin_memory=True, num_workers=num_workers)

loss_weights = [1, 2]
#endregion

#region training
save_model_dir = f'/tmp2/2021_6_8/{data_version}'
train_times = 1
train_type = 'pre_trained' #scratch, pre_trained

for train_time in range(train_times):
    for model_name in ['medical_net_resnet50']:

        model = get_model(model_name, num_class=num_class)

        if model_name == 'Cls_3d':
            if train_type == 'pre_trained':
                model_file = '/disk1/Models_Genesis/Genesis_Chest_CT.pt'
                state_dict = torch.load(model_file, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)

        #region medical net
        if model_name == 'medical_net_resnet34':
            if train_type == 'pre_trained':
                model_file ='/disk1/MedicalNet_pytorch_files/pretrain/resnet_34.pth'
                state_dict = torch.load(model_file, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
        if model_name == 'medical_net_resnet50':
            if train_type == 'pre_trained':
                model_file ='/disk1/MedicalNet_pytorch_files/pretrain/resnet_50.pth'
                state_dict = torch.load(model_file, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)

        if model_name == 'medical_net_resnet101':
            if train_type == 'transfer_learning':
                model_file ='/disk1/MedicalNet_pytorch_files/pretrain/resnet_101.pth'
                state_dict = torch.load(model_file, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)

        #endregion

        '''        
        if model_name == 'ModelsGenesis':
            if train_type == 'transfer_learning':
                model_file = '/disk1/Models_Genesis/Genesis_Chest_CT.pt'
                state_dict = torch.load(model_file, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
        '''


        loss_weights = torch.FloatTensor(loss_weights)
        if torch.cuda.device_count() > 0:
            loss_weights = loss_weights.cuda()
        label_smoothing = 0
        if label_smoothing > 0:
            from libs.neural_networks.loss.my_label_smoothing import LabelSmoothLoss
            criterion = LabelSmoothLoss(class_weight=loss_weights, smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(weight=loss_weights)

        optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=0.001)
        # from libs.neural_networks.my_optimizer import Lookahead
        # optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.3)
        epochs_num = 15

        train(model,
              loader_train=loader_train,
              criterion=criterion, optimizer=optimizer, scheduler=scheduler,
              epochs_num=epochs_num, amp=True, log_interval_train=10,
              loader_valid=loader_valid,
              loader_test=loader_test,
              save_model_dir=os.path.join(save_model_dir, model_name, str(train_time)),
              )

    del model
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()

#endregion

print('OK')

