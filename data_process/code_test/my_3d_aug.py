import torch
import torchio as tio
from torch.utils.data import DataLoader

# Images may also be created using PyTorch tensors or NumPy arrays
tensor_4d = torch.rand(4, 100, 100, 100)
# tensor_4d = torch.ones(4, 10, 10, 10)
tensor_4d = tensor_4d * 255
subject_c = tio.Subject(
    oct=tio.ScalarImage(tensor=tensor_4d),
)

# 10

subjects_list = [subject_c]

transform = tio.Compose([
    tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    }, p=0.75,),
    # tio.RandomGamma(log_gamma=(-0.3, 0.3)),
    tio.RandomAffine(),
    tio.RandomFlip(axes=1, flip_probability=0.5),
    tio.RandomNoise(std=(0, 0.1)),
    tio.Crop(cropping=(10, 10, 10, 10, 10, 10)),
    tio.Pad(padding=(10, 10, 10, 10, 10, 10)),
    tio.Resample((1, 2, 2)),
    tio.RescaleIntensity((0, 255))]
)


subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)
training_loader = DataLoader(subjects_dataset, batch_size=4, num_workers=4)

# Training epoch
for subjects_batch in training_loader:
    inputs = subjects_batch['oct'][tio.DATA]
    print('min:', torch.min(inputs), 'max:', torch.max(inputs))
    # print(inputs)
    print(inputs.shape)

print('ok')