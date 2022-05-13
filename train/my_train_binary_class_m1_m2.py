import warnings
warnings.filterwarnings("ignore")
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse

#region command line parameters and import libraries
parser = argparse.ArgumentParser()
parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1')
parser.add_argument('--task_type', default='3D_OCT_DME_M1_M2')
parser.add_argument('--data_version', default='v1')
parser.add_argument('--model_name', default='ModelsGenesis') #cls_3d, ModelsGenesis, medical_net_resnet50
parser.add_argument('--drop_prob', default=0)
parser.add_argument('--image_shape',  nargs='+', type=int, default=(128, 128)) #(64, 64), (96,96), (128,128)
parser.add_argument('--random_add', default=8)
parser.add_argument('--random_crop_h', default=9)
parser.add_argument('--random_noise', default=0.2)

parser.add_argument('--pos_weight', nargs='+', type=float, default=(0.85,))  #cost sensitive learning, weighted binary cross entropy
parser.add_argument('--label_smoothing', default=0)
parser.add_argument('--amp', action='store_true', default=False)  #AUTOMATIC MIXED PRECISION
#recommend num_workers = the number of gpus * 4, when debugging it should be set to 0.
parser.add_argument('--num_workers', default=4)
parser.add_argument('--batch_size_train', default=32)
parser.add_argument('--batch_size_valid', default=32)
parser.add_argument('--weight_decay', default=0)
parser.add_argument('--epochs_num', default=20)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--step_size', default=5)
parser.add_argument('--gamma', default=0.3)
parser.add_argument('--log_interval_train', default=10)
parser.add_argument('--save_model_dir', default='/tmp2/2022_4_24/binary_classifier_m1_m2/')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
from imgaug import augmenters as iaa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from libs.dataset.my_dataset_torchio import Dataset_CSV_train, Dataset_CSV_test
from torch.utils.data import DataLoader
from libs.neural_networks.model.my_get_model import get_model
from libs.neural_networks.helper.my_train_binary_class import train

# print(args)
#endregion

#region dataset
path_csv = Path(__file__).resolve().parent.parent / 'datafiles' / args.data_version
csv_train = path_csv / f'{args.task_type}_train.csv'
csv_valid = path_csv / f'{args.task_type}_valid.csv'
csv_test = path_csv / f'{args.task_type}_test.csv'

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

ds_train = Dataset_CSV_train(csv_file=csv_train,  image_shape=args.image_shape,
                             random_crop_h=args.random_crop_h, random_noise=args.random_noise,
                             imgaug_iaa=imgaug_iaa)
loader_train = DataLoader(ds_train, batch_size=args.batch_size_train, shuffle=True,
                          pin_memory=True, num_workers=args.num_workers)
ds_valid = Dataset_CSV_test(csv_file=csv_valid, image_shape=args.image_shape,
                            depth_start=0, depth_interval=2)
loader_valid = DataLoader(ds_valid, batch_size=args.batch_size_valid,
                           pin_memory=True, num_workers=args.num_workers)
ds_test = Dataset_CSV_test(csv_file=csv_test, image_shape=args.image_shape,
                           depth_start=0, depth_interval=2)
loader_test = DataLoader(ds_test, batch_size=args.batch_size_valid,
                            pin_memory=True, num_workers=args.num_workers)

#endregion

#region load model transfer from m0 vs m1 and m2
path_trained_models = Path(__file__).resolve().parent.parent / 'trained_models' / '2022_4_28_64_64' / 'binary_class_m0_m1m2'
if args.model_name == 'cls_3d':
    model_file = path_trained_models / 'cls_3d.pth'
if args.model_name == 'medical_net_resnet50':
    model_file = path_trained_models / 'medical_net_resnet50.pth'
model = get_model(args.model_name, num_class=1, model_file=model_file, drop_prob=args.drop_prob) #binary classification

#endregion


#region training
pos_weight = torch.FloatTensor(torch.tensor(args.pos_weight))
if torch.cuda.is_available():
    pos_weight = pos_weight.cuda()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
#from libs.neural_networks.optimizer_obsoleted.my_optimizer import Lookahead
# optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

train(model, loader_train=loader_train, criterion=criterion, optimizer=optimizer,
      activation='sigmoid', label_smoothing=args.label_smoothing,
      scheduler=scheduler, epochs_num=args.epochs_num, log_interval_train=args.log_interval_train,
      loader_valid=loader_valid, loader_test=loader_test, amp=args.amp,
      save_model_dir=os.path.join(args.save_model_dir, f'{args.task_type}_{args.data_version}', args.model_name)
      )

# del model
# if torch.cuda.device_count() > 0:
#         torch.cuda.empty_cache()

#endregion


print('OK')

