import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from libs.dataset.my_dataset import Dataset_CSV_3d
from torch.utils.data import DataLoader
import pandas as pd
import shutil

dir_original = '/disk1/3D_OCT_DME/original/'
dir_preprocess = '/disk1/3D_OCT_DME/preprocess/64_64_64/'
dir_dest = '/tmp2/3D_OCT_DME/confusion_files1/'

# data_version = 'v1_topocon_zeiss_64_64_64'
filename_csv = os.path.join(os.path.abspath('..'),
                'datafiles', 'v1_topocon_64_64_64', f'3D_OCT_DME_split_patid_test.csv')

from libs.neural_networks.ModelsGenesis.unet3d import UNet3D_classification
model = UNet3D_classification(n_class=2)
model_file = '/tmp2/v1_topocon_64_64_64/ModelsGenesis/0/epoch6.pth'
state_dict = torch.load(model_file, map_location='cpu')
model.load_state_dict(state_dict, strict=False)

batch_size_valid = 32
image_shape = (64, 64)
ds_valid = Dataset_CSV_3d(csv_file=filename_csv, image_shape=image_shape, test_mode=True)
loader_valid = DataLoader(ds_valid, batch_size=batch_size_valid,
                          num_workers=4)

from libs.helper.my_predict import predict
(probs, lables_pd) = predict(model, loader_valid)

df = pd.read_csv(filename_csv)
(image_files, labels) = list(df['images']), list(df['labels'])

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels, lables_pd)
print(cf)

for image_file, label_gt, prob in \
        zip(image_files, labels, probs):
    label_predict = prob.argmax()
    if label_gt != label_predict:
        image_file = image_file.replace(dir_preprocess, dir_original)
        dir_base = os.path.dirname(image_file)
        for dir_path, subpaths, files in os.walk(dir_base, False):
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


print('OK')

