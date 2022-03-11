# **ImageNet-Patch**

The demo code for the application of the generated patches on a batch from the Imagenet dataset.


# Documentation

Paper -> Preprint available at https://arxiv.org/abs/2203.04412

Tutorial -> link file example.py, a detailed example of how to use ImageNet-Patch in your model predictions

# How can I load the patches?

You can find the patches saved in assets/patches.gz in the form (patches, targets),
where patches and targets are pytorch tensors respectively with shape (10, 3, 224, 224) and (10,).
To load them you can simply run the following code snippet and select your favourite one :)

```python
# Load the patches
with gzip.open(os.path.join(os.getcwd(), "assets/patches.gz"), 'rb') as f:
    patches, targets = pickle.load(f)
patch = patches[1]  # get the patch with id=1
target = targets[1]
```

Once a patch is selected it can be used 
instantiating the ApplyPatch module in the preprocessing stack, just before the normalizer!
To do that you have to specify the patch tensor and the affine transformations
ranges (translation_range, rotation_range, scale_range), for which the parameters will be random sampled when an image is taken.


```python
from transforms.apply_patch import ApplyPatch
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose

# Instantiate the ApplyPatch layer, setting patch, target and transforms
apply_patch = ApplyPatch(patch,
                         translation_range=(.2, .2),    # translation fraction wrt image dimensions
                         rotation_range=45,             # maximum absolute value of the rotation in degree
                         scale_range=(0.5, 1)           # scale range wrt image dimensions
                         )

# Build the preprocessing stack with ApplyPatch before the normalization step
preprocess = Compose([Resize(256), CenterCrop(224), ToTensor(),
                      apply_patch,
                      Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                      ])
```

# Citation

bibtek arxiv