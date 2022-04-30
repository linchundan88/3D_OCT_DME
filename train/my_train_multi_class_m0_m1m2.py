import warnings
warnings.filterwarnings("ignore")
import os
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)
import argparse

#region command line parameters and import libraries
parser = argparse.ArgumentParser()
parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1')
parser.add_argument('--task_type', default='3D_OCT_DME_M0_M1M2')
parser.add_argument('--data_version', default='v3')
parser.add_argument('--sampling_class_weights', default=(1, 2))  #dynamic resampling
parser.add_argument('--model_name', default='cls_3d')   # cls_3d, medical_net_resnet50
parser.add_argument('--drop_prob', default=0)
parser.add_argument('--pre_trained', default=True)  #initialize weights from pre-trained model
parser.add_argument('--image_shape', default=(64, 64))
parser.add_argument('--random_add', default=8)
parser.add_argument('--random_crop_h', default=9)
parser.add_argument('--random_noise', default=0.2)

parser.add_argument('--loss_weights', default=(1, 2))  #cost sensitive learning, weighted cross entropy
parser.add_argument('--label_smoothing', default=0)
parser.add_argument('--amp', default=False)  #AUTOMATIC MIXED PRECISION
#recommend num_workers = the number of gpus * 4, when debugging it should be set to 0.
parser.add_argument('--num_workers', default=4)
parser.add_argument('--batch_size_train', default=32)
parser.add_argument('--batch_size_valid', default=32)
parser.add_argument('--weight_decay', default=0)
parser.add_argument('--epochs_num', default=15)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--step_size', default=3)
parser.add_argument('--gamma', default=0.3)
parser.add_argument('--log_interval_train', default=10)

parser.add_argument('--save_model_dir', default='/tmp2/2021_8_24/')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
import pandas as pd
from imgaug import augmenters as iaa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from libs.dataset.my_dataset_torchio import Dataset_CSV_train, Dataset_CSV_test
from torch.utils.data import DataLoader
from libs.neural_networks.model.my_get_model import get_model
from libs.neural_networks.helper.my_train_multi_class import train

#endregion

#region dataset
csv_train = Path(__file__).parent.parent / 'datafiles' / args.data_version / f'{args.task_type}_train.csv'
csv_valid = Path(__file__).parent.parent / 'datafiles' / args.data_version / f'{args.task_type}_valid.csv'
csv_test = Path(__file__).parent.parent / 'datafiles' / args.data_version / f'{args.task_type}_test.csv'

imgaug_iaa = iaa.Sequential([
    # iaa.Fliplr(0.5),

    # iaa.GaussianBlur(sigma=(0.0, 0.3)),
    # iaa.MultiplyBrightness(mul=(0.8, 1.2)),
    # iaa.contrast.LinearContrast((0.8, 1.2)),

    iaa.Sometimes(0.9, iaa.Add((-args.random_add, args.random_add))),
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
for label in df['labels']:
    sampling_weights.append(args.sampling_class_weights[label])
sampler = WeightedRandomSampler(weights=sampling_weights, num_samples=len(df))


ds_train = Dataset_CSV_train(csv_file=csv_train,  image_shape=args.image_shape,
                             random_crop_h=args.random_crop_h, random_noise=args.random_noise,
                             imgaug_iaa=imgaug_iaa)
loader_train = DataLoader(ds_train, batch_size=args.batch_size_train,
                          sampler=sampler, pin_memory=True, num_workers=args.num_workers)
ds_valid = Dataset_CSV_test(csv_file=csv_valid, image_shape=args.image_shape,
                            depth_start=0, depth_interval=2)
loader_valid = DataLoader(ds_valid, batch_size=args.batch_size_valid,
                           pin_memory=True, num_workers=args.num_workers)
ds_test = Dataset_CSV_test(csv_file=csv_test, image_shape=args.image_shape,
                           depth_start=0, depth_interval=2)
loader_test = DataLoader(ds_test, batch_size=args.batch_size_valid,
                            pin_memory=True, num_workers=args.num_workers)

#endregion

#region define model
if args.pre_trained:
    if args.model_name == 'cls_3d':
        model_file = '/disk1/Models_Genesis/Genesis_Chest_CT.pt'
    if args.model_name == 'medical_net_resnet34':
        model_file ='/disk1/MedicalNet_pytorch_files/pretrain/resnet_34.pth'
    if args.model_name == 'medical_net_resnet50':
        model_file ='/disk1/MedicalNet_pytorch_files/pretrain/resnet_50.pth'
    if args.model_name == 'medical_net_resnet101':
        model_file ='/disk1/MedicalNet_pytorch_files/pretrain/resnet_101.pth'
    if args.model_name == 'ModelsGenesis' :
        model_file = '/disk1/Models_Genesis/Genesis_Chest_CT.pt'

else:
    model_file = None

model = get_model(args.model_name, num_class=num_class, model_file=model_file, drop_prob=args.drop_prob)

#endregion


#region training
loss_weights = torch.FloatTensor(args.loss_weights)
if torch.cuda.device_count() > 0:
    loss_weights = loss_weights.cuda()
if args.label_smoothing > 0:
    from libs.neural_networks.loss.my_label_smoothing import LabelSmoothLoss
    criterion = LabelSmoothLoss(class_weight=loss_weights, smoothing=args.label_smoothing)
else:
    criterion = nn.CrossEntropyLoss(weight=loss_weights, reduction='mean')

optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
# from libs.neural_networks.optimizer_obsoleted.my_optimizer import Lookahead
# optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

train(model, loader_train=loader_train, criterion=criterion, optimizer=optimizer, activation='softmax',
      scheduler=scheduler, epochs_num=args.epochs_num, log_interval_train=args.log_interval_train,
      loader_valid=loader_valid, loader_test=loader_test, amp=args.amp,
      save_model_dir=os.path.join(args.save_model_dir, f'{args.task_type}_{args.data_version}', args.model_name)
      )

# del model
# if torch.cuda.device_count() > 0:
#         torch.cuda.empty_cache()

#endregion


print('OK')

