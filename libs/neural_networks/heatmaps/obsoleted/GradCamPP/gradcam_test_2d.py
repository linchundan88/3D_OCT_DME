
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from libs.neural_networks.heatmaps.obsoleted.GradCamPP import GradCAM, GradCAMpp
import pretrainedmodels

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#region load model

# model_name = 'xception'
# model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# image_shape = (299, 299)
# model_type = 'others'
# layer_name = 'conv4'

model_name = 'inceptionresnetv2'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
image_shape = (299, 299)
model_type = 'others'
layer_name = 'conv2d_7b'

# model_name = 'inceptionv3'
# model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# image_shape = (299, 299)
# model_type = 'others'
# layer_name = 'Mixed_7c'

model = model.to(device).eval()

#endregion

#region laod image
img_path = os.path.join(os.path.abspath(''), 'cat.jpeg')

pil_img = PIL.Image.open(img_path)

torch_img = transforms.Compose([
    transforms.Resize(image_shape),
    transforms.ToTensor()
])(pil_img).to(device)
normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
#endregion


logits = model(normed_torch_img)
# probabilities = F.softmax(logits, dim=1).data.squeeze()
probabilities = F.softmax(logits, dim=1)
_, pred = torch.max(probabilities, 1)

cam_type = 'gradcam'
if cam_type == 'gradcam':
    gradcam = GradCAM.from_config(model_type=model_type, arch=model, layer_name=layer_name)
    mask, logit = gradcam(normed_torch_img, class_idx=pred)

if cam_type == 'gradcampp':
    gradcampp = GradCAMpp.from_config(model_type=model_type, arch=model, layer_name=layer_name)
    for _ in range(3):
        mask, logit = gradcampp(normed_torch_img, class_idx=pred)

from matplotlib.pyplot import show
mask = mask.cpu().numpy()
mask = np.squeeze(mask)

# imshow(mask, alpha=0.5, cmap='jet', bbox_inches='tight')
# show()

import matplotlib.pyplot as plt
img = plt.imshow(mask, alpha=0.5, cmap='jet')

plt.imsave("foo.png", mask, format="png", cmap="jet")

plt.axis("off")   # turns off axes
plt.axis("tight")  # gets rid of white border
plt.axis("image")  # square up the image instead of filling the "figure" space
plt.savefig("test.png", bbox_inches='tight', pad_inches=0)
show()

print('OK')