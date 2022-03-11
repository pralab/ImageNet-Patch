import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import ImageFolder

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, \
    Normalize
import torchvision.models as models

from utils.visualization import imshow
from utils.utils import set_all_seed
from utils.utils import target_transforms

from transforms.apply_patch import ApplyPatch

import gzip
import pickle

import os


set_all_seed(0)

# Choose an integer in the range 0-9 to select the patch
patch_id = 1

# dictionary with the ImageNet label names
with open(os.path.join(os.getcwd(), "assets/imagenet1000_clsidx_to_labels.txt")) as f:
    target_to_classname = eval(f.read())

# Load the patches
with gzip.open(os.path.join(os.getcwd(), "assets/imagenet_patch.gz"), 'rb') as f:
    imagenet_patch = pickle.load(f)
patches, targets, info = imagenet_patch
patch = patches[patch_id]      # get the patch with id=1

print(f"Target class: {target_to_classname[targets[patch_id].item()].split(',')[0]}")

# Instantiate the ApplyPatch module setting the patch and the affine transformation that will be applied
apply_patch = ApplyPatch(patch, patch_size=info['patch_size'],
                         translation_range=(.2, .2),    # translation fraction wrt image dimensions
                         rotation_range=45,             # maximum absolute value of the rotation in degree
                         scale_range=(0.7, 1)           # scale range wrt image dimensions
                         )

# For convenience the preprocessing steps are splitted to compute also the clean predictions
normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
patch_normalizer = Compose([apply_patch, normalizer])

# Load the data
preprocess = Compose([Resize(256), CenterCrop(224), ToTensor()])    # ensure images are 224x224
dataset = ImageFolder(os.path.join(os.getcwd(), "assets/data"),
                      transform=preprocess,
                      target_transform=target_transforms)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
x, y = next(iter(data_loader))  # load a mini-batch
x_clean = normalizer(x)
x_adv = patch_normalizer(x)

imshow(torchvision.utils.make_grid(x_adv.cpu().detach(), nrow=5), transforms=patch_normalizer)

# Load model
model = models.alexnet(pretrained=True)
model.eval()

# Test the model with the clean images
output_clean = model(x_clean)
clean_predictions = torch.argmax(output_clean, dim=1).cpu().detach().numpy()

# Test the model with the images corrupted by the patch
output_adv = model(x_adv)
adv_predictions = torch.argmax(output_adv, dim=1).cpu().detach().numpy()

print("\nPredictions:")
for true_label, clean_pred, adv_pred in list(zip(y, clean_predictions, adv_predictions)):
    print(f"True label: {target_to_classname[true_label.item()].split(',')[0]} -> "
          f"Clean: {target_to_classname[clean_pred].split(',')[0]} -> "
          f"Adv.: {target_to_classname[adv_pred].split(',')[0]}")
