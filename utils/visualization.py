from torchvision.transforms import Normalize
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import os

class InvNormalize(Normalize):
    def __init__(self, normalizer):
        inv_mean = [-mean / std for mean, std in list(zip(normalizer.mean, normalizer.std))]
        inv_std = [1 / std for std in normalizer.std]
        super().__init__(inv_mean, inv_std)

def _tensor_to_show(img, transforms=None):
    if transforms is not None:
        for transform in transforms.transforms:
            if isinstance(transform, Normalize):
                normalizer = transform
                break
        inverse_transform = InvNormalize(normalizer)
        img = inverse_transform(img)

    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg

def imshow(img, transforms=None, figsize=(10, 20)):
    npimg = _tensor_to_show(img, transforms)
    plt.figure(figsize=figsize)
    plt.imshow(npimg, interpolation=None)

upp_l_x = 224 // 2 - 50 // 2
upp_l_y = 224 // 2 - 50 // 2
bott_r_x = 224 // 2 + 50 // 2
bott_r_y = 224 // 2 + 50 // 2

# dictionary with the ImageNet label names
with open(os.path.join(os.getcwd(), "assets/imagenet1000_clsidx_to_labels.txt")) as f:
    target_to_classname = eval(f.read())

def show_imagenet_patch(patches, targets):
    fig = plt.figure(figsize=(10, 5))

    for i, (patch, target) in enumerate(list(zip(patches, targets))):
        patch = np.transpose(patch, (1, 2, 0))
        patch = patch[upp_l_x:bott_r_x, upp_l_y:bott_r_y, :]

        plt.subplot(2, 5, i+1)
        plt.title(target_to_classname[int(target.item())].split(",")[0])
        plt.imshow(patch)
        plt.axis('off')
    plt.show()


def show_batch_with_patch(x, transforms=None, figsize=(10, 20)):
    imshow(make_grid(x.cpu().detach(), nrow=5),
           transforms=transforms, figsize=figsize)
    plt.axis('off')
    plt.show()


def plot_patch_predictions(x_clean, x_adv, clean_pred, adv_pred,
                           true_label, target, figsize=(5, 20),
                           normalizer=None):
    N_IMAGES = x_clean.shape[0]

    fig, ax = plt.subplots(2, N_IMAGES, figsize=figsize)

    if normalizer is not None:
        inverse_transform = InvNormalize(normalizer)
        x_clean = inverse_transform(x_clean)
        x_adv = inverse_transform(x_adv)
    x_clean_img = np.transpose(x_clean.detach().cpu().numpy(), (0, 2, 3, 1))
    x_adv_img = np.transpose(x_adv.detach().cpu().numpy(), (0, 2, 3, 1))

    # plot clean samples in first row
    for j in range(N_IMAGES):
        true_j = true_label[j]
        true = target_to_classname[true_j].split(",")[0]
        clean_j = clean_pred[j]
        clean = target_to_classname[clean_j].split(",")[0]
        ax[0, j].imshow(x_clean_img[j])
        ax[0, j].set_title(f"Pred.: {clean}\nTrue: {true}")
        ax[0, j].set_xticks([])
        ax[0, j].set_yticks([])

    for j in range(N_IMAGES):
        ax[1, j].imshow(x_adv_img[j])
        p_i = adv_pred[j].item()
        p = target_to_classname[p_i].split(",")[0]
        true_i = true_label[j].item()
        true = target_to_classname[true_i].split(",")[0]

        if p_i == target:
            color = 'r'
        elif p_i != true_i:
            color = 'b'
        else:
            color = 'g'
        ax[1, j].set_title(f"Pred.: {p}", color=color)
        ax[1, j].set_xticks([])
        ax[1, j].set_yticks([])

    ax[1, 0].set_ylabel(target_to_classname[target].split(",")[0])
    ax[0, 0].set_ylabel("Clean")
    plt.show()