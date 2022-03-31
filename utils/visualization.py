from torchvision.transforms import Normalize
import numpy as np
import matplotlib.pyplot as plt

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

def imshow(img, transforms=None):
    npimg = _tensor_to_show(img, transforms)
    plt.figure()
    plt.imshow(npimg, interpolation=None)
    plt.show()

def plot_imagenet_patch(patches, targets):
    upp_l_x = 224 // 2 - 50 // 2
    upp_l_y = 224 // 2 - 50 // 2
    bott_r_x = 224 // 2 + 50 // 2
    bott_r_y = 224 // 2 + 50 // 2

    target_to_classname = {"968": 'cup',
                           "804": 'soap dispenser',
                           "806": 'sock',
                           "923": 'plate',
                           "954": 'banana',
                           "585": 'hair spray',
                           "546": 'electric guitar',
                           "513": 'brass',
                           "878": 'typewriter keyboard',
                           "487": 'cellphone'
                           }

    fig = plt.figure(figsize=(10, 5))

    for i, (patch, target) in enumerate(list(zip(patches, targets))):
        patch = np.transpose(patch, (1, 2, 0))
        patch = patch[upp_l_x:bott_r_x, upp_l_y:bott_r_y, :]

        plt.subplot(2, 5, i+1)
        plt.title(target_to_classname[str(int(target.item()))])
        plt.imshow(patch)
        plt.axis('off')
    plt.show()
