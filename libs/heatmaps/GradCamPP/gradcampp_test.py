
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from libs.NeuralNetworks.Heatmaps.GradCamPP.gradcampp import GradCAM, GradCAMpp
import pretrainedmodels

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#region load model

# model = models.vgg16(pretrained=True)
# image_shape = (299, 299)
# model_type = 'vgg'
# layer_name = features_29

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
img_path = os.path.join(os.path.abspath('.'), 'cat.jpeg')

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
    # mask, logit = gradcam(normed_torch_img)  #class_idx None choose top1 class
if cam_type == 'gradcampp':
    gradcampp = GradCAMpp.from_config(model_type=model_type, arch=model, layer_name=layer_name)
    for _ in range(3):
        mask, logit = gradcampp(normed_torch_img, class_idx=pred)

from matplotlib.pyplot import imshow, show
mask = mask.cpu().numpy()
mask = np.squeeze(mask)
imshow(mask, alpha=0.5, cmap='jet')
show()


print('OK')