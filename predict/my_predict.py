import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch.utils.data import DataLoader
import pandas as pd

dir_original = '/disk1/3D_OCT_DME/original/'
dir_preprocess = '/disk1/3D_OCT_DME/preprocess/128_128_128/'
dir_dest = '/tmp2/3D_OCT_DME/confusion_files1/'

filename_csv = os.path.join(os.path.abspath('..'),
                'datafiles', 'v1_topocon_128_128_128', f'3D_OCT_DME_split_patid_test.csv')
num_class = 2

model_name = 'ModelsGenesis'

#region define model
if model_name == 'ModelsGenesis':
    from libs.neural_networks.model.cls_3d import Cls_3d
    model = Cls_3d(n_class=num_class)

if model_name == 'medical_net_resnet34':
    from libs.neural_networks.model.MedicalNet.resnet import resnet34, Resnet3d_cls
    base_model = resnet34(output_type='classification')
    model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='BasicBlock', add_dense1=True)
if model_name == 'medical_net_resnet50':
    from libs.neural_networks.model.MedicalNet.resnet import resnet50, Resnet3d_cls
    base_model = resnet50(output_type='classification')
    model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='Bottleneck', add_dense1=True)
if model_name == 'medical_net_resnet101':
    from libs.neural_networks.model.MedicalNet.resnet import resnet101, Resnet3d_cls
    base_model = resnet101(output_type='classification')
    model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='Bottleneck', add_dense1=True)

#endregion

model_file = '/tmp2/2020_3_11/v1_topocon_128_128_128/ModelsGenesis/0/epoch7.pth'
state_dict = torch.load(model_file, map_location='cpu')
model.load_state_dict(state_dict, strict=False)


from libs.dataset.my_dataset_torchio import Dataset_CSV_test
batch_size_valid = 32
image_shape = (64, 64)
ds_valid = Dataset_CSV_test(csv_file=filename_csv, image_shape=image_shape,
                            depth_start=0, depth_interval=2, test_mode=True)
loader_valid = DataLoader(ds_valid, batch_size=batch_size_valid, pin_memory=True,  num_workers=4)

from libs.neural_networks.helper.my_predict import predict
(probs, lables_pd) = predict(model, loader_valid)

df = pd.read_csv(filename_csv)
(image_files, labels) = list(df['images']), list(df['labels'])

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels, lables_pd)
print(cf)


'''

for image_file, label_gt, prob in \
        zip(image_files, labels, probs):
    label_predict = prob.argmax()
    if label_gt != label_predict:
        image_file = image_file.replace(dir_preprocess, dir_original)
        dir_base = os.path.dirname(image_file)
        for dir_path, _, files in os.walk(dir_base, False):
            for f in files:
                file_name_full = os.path.join(dir_path, f)
                _, filaname = os.path.split(file_name_full)
                filebase, file_ext = os.path.splitext(filaname)
                if file_ext.lower() in ['.jpeg', '.jpg', '.png']:
                    filename_tmp = file_name_full.replace(dir_original, '')
                    if label_gt == 0:
                        img_dest = os.path.join(dir_dest, f'{label_gt}_{label_predict}',
                                                f'prob_{int(prob[0] * 100)}', filename_tmp)
                    if label_gt == 1:
                        img_dest = os.path.join(dir_dest, f'{label_gt}_{label_predict}',
                                                f'prob_{int(prob[1] * 100)}', filename_tmp)

                    print(img_dest)
                    os.makedirs(os.path.dirname(img_dest), exist_ok=True)
                    shutil.copy(file_name_full, img_dest)

'''


