#**ImageNet-Patch**

The demo code for the application of the generated patches on a batch from the Imagenet dataset.

Preprint available at https://arxiv.org/abs/2203.04412

The patches are saved in assets/patches.gz in the form (patches, targets),
where patches and targets are pytorch tensors respectively with shape (10, 3, 224, 224) and (10,).

Once a patch with its associated target class is selected it can be used simply 
instantiating the ApplyPatch module in the preprocessing stack, just before the normalizer!
To instantiate this module we must specify the patch, the target class and the affine transformation
ranges (translation_range, rotation_range, scale_range), for which the parameters will be random sampled when an image is taken.
