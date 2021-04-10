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
from libs.helper.my_train import train


#region prepare dataset
task_type = '3D_OCT_DME'
image_shape = (64, 64)
data_version = 'v1_topocon_128_128_128'
csv_train = os.path.join(os.path.abspath('..'),
                'datafiles', data_version, f'{task_type}_split_patid_train.csv')
csv_valid = os.path.join(os.path.abspath('..'),
                'datafiles', data_version, f'{task_type}_split_patid_valid.csv')
csv_test = os.path.join(os.path.abspath('..'),
                'datafiles', data_version, f'{task_type}_split_patid_test.csv'.format(data_version))

'''
v1_topocon_128_128_128

3327
0 3039
1 288
705
0 645
1 60
731
0 682
1 49

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

class_weights = [1, 2.5]

from torch.utils.data.sampler import WeightedRandomSampler
list_class_samples = []
for label in range(num_class):
    list_class_samples.append(len(df[df['labels'] == label]))
# sample_class_weights = 1 / np.power(list_class_samples, 0.5)
sample_class_weights = [1, 3]
sample_weights = []
for label in df['labels']:
    sample_weights.append(sample_class_weights[label])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(df))

batch_size_train, batch_size_valid = 32, 32
ds_train = Dataset_CSV_train(csv_file=csv_train, image_shape=image_shape)
loader_train = DataLoader(ds_train, batch_size=batch_size_train,
                          sampler=sampler, num_workers=4)
ds_valid = Dataset_CSV_test(csv_file=csv_valid, image_shape=image_shape,
                            depth_start=0, depth_interval=2)
loader_valid = DataLoader(ds_valid, batch_size=batch_size_valid,
                           num_workers=4)
ds_test = Dataset_CSV_test(csv_file=csv_test, image_shape=image_shape,
                           depth_start=0, depth_interval=2)
loader_test = DataLoader(ds_test, batch_size=batch_size_valid,
                        num_workers=4)
#endregion

#region training
save_model_dir = f'/tmp2/2020_3_11_wo_pretrain/{data_version}'
train_times = 1
train_type = 'scratch' #scratch, pre_trained

for train_time in range(train_times):
    for model_name in ['medical_net_resnet50']:
        if model_name == 'ModelsGenesis':
            '''
            from libs.neural_networks.ModelsGenesis.unet3d import UNet3D, TargetNet
            base_model = UNet3D()
            if train_type == 'transfer_learning':
                model_file = '/disk1/Models_Genesis/Genesis_Chest_CT.pt'
                state_dict = torch.load(model_file, map_location='cpu')
                base_model.load_state_dict(state_dict, strict=False)
            model = TargetNet(base_model, n_class=num_class)
            '''
            from libs.neural_networks.ModelsGenesis.unet3d import UNet3D_classification
            model = UNet3D_classification(n_class=num_class)
            if train_type == 'pre_trained':
                model_file = '/disk1/Models_Genesis/Genesis_Chest_CT.pt'
                state_dict = torch.load(model_file, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)

        #region medical net
        if model_name == 'medical_net_resnet34':
            from libs.neural_networks.MedicalNet.resnet import resnet34, Resnet3d_cls
            base_model = resnet34(output_type='classification')
            if train_type == 'pre_trained':
                model_file ='/disk1/MedicalNet_pytorch_files/pretrain/resnet_34.pth'
                state_dict = torch.load(model_file, map_location='cpu')
                base_model.load_state_dict(state_dict, strict=False)
            model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='BasicBlock', add_dense1=True)
        if model_name == 'medical_net_resnet50':
            from libs.neural_networks.MedicalNet.resnet import resnet50, Resnet3d_cls
            base_model = resnet50(output_type='classification')
            if train_type == 'pre_trained':
                model_file ='/disk1/MedicalNet_pytorch_files/pretrain/resnet_50.pth'
                state_dict = torch.load(model_file, map_location='cpu')
                base_model.load_state_dict(state_dict, strict=False)
            model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='Bottleneck', add_dense1=True)
        if model_name == 'medical_net_resnet101':
            from libs.neural_networks.MedicalNet.resnet import resnet101, Resnet3d_cls
            base_model = resnet101(output_type='classification')
            if train_type == 'transfer_learning':
                model_file ='/disk1/MedicalNet_pytorch_files/pretrain/resnet_101.pth'
                state_dict = torch.load(model_file, map_location='cpu')
                base_model.load_state_dict(state_dict, strict=False)
            model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='Bottleneck', add_dense1=True)
        #endregion

        #region model_3d  [10, 18, 34, 50, 101, 152, 200]
        from libs.neural_networks.model_3d.resnet import generate_model
        if model_name == 'resnet18':
            model = generate_model(model_depth=18, n_classes=num_class, n_input_channels=1)
        if model_name == 'resnet34':
            model = generate_model(model_depth=32, n_classes=num_class, n_input_channels=1)
        if model_name == 'resnet50':
            model = generate_model(model_depth=50, n_classes=num_class, n_input_channels=1)
        if model_name == 'resnet101':
            model = generate_model(model_depth=101, n_classes=num_class, n_input_channels=1)

        #endregion

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_class_weights = torch.FloatTensor(class_weights)
        if torch.cuda.device_count() > 0:
            model.to(device)
            loss_class_weights = loss_class_weights.cuda()
        label_smoothing = 0
        if label_smoothing > 0:
            from libs.loss.my_label_smoothing import LabelSmoothLoss
            criterion = LabelSmoothLoss(class_weight=loss_class_weights, smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(weight=loss_class_weights)

        optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=0.001)
        # from libs.neural_networks.my_optimizer import Lookahead
        # optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)

        scheduler = StepLR(optimizer, step_size=4, gamma=0.3)
        epochs_num = 20

        train(model,
              loader_train=loader_train,
              criterion=criterion, optimizer=optimizer, scheduler=scheduler,
              epochs_num=epochs_num, log_interval_train=10,
              loader_valid=loader_valid, loader_test=loader_test,
              save_model_dir=os.path.join(save_model_dir, model_name, str(train_time))
              )

    del model
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()

#endregion

print('OK')

