"""
REQUISITI DEL REPO
-le patch
-uno script che data un'immagine applica le patch con le nostre trasformazioni
-ovviamente file requirements e readme
-(opzionale) qualche immagine di esempio
-(opzionale ma meglio averlo) un "example.py" pronto da far girare che fa anche il test su un modello e qualche immagine

IMPLEMENTAZIONE
-patch: tensori? np array? png? (magari png così si possono sia guardare che usare nel codice) BOTH
    id target label, (x,y) patches.gz -> in png 878-banana.png
-trasformazioni: usiamo i seed o salviamo i parametri? Meglio la seconda secondo me, ma devo capire come fare
-applicazione patch su img: funzione/metodo apply_patch (img, patch, transform)
    includere codice delle transform a prescindere
-immagine di esempio: si può mettere quella del paper con le predictions?
-example.py: devo fare la pappa pronta per testbed e robustbench o posso mettere solo modello torchvision comodo?

DUBBI
Devo usare lo stesso indexing del paper? Prendere il test set di robustbench, togliere i sample della classe target ecc...

- Nel readme dire di procurarsi validation di imagenet e poi usare utility per idx di robustbench

- classe che data img restituisce immagine con patch

carica modello
carica dati (batch) -> da mettere in cartella assets/dati da caricare come dataset ImageFolder
applica patch
test img con patch


<<<esempio minimale>>>
istanzia ApplyPatch
inserisci nelle transform (before normalization)

"""
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, \
    Normalize
import torchvision.models as models
import torch

from utils.visualization import imshow
from utils.utils import set_all_seed
from utils.utils import target_transforms

from transforms.apply_patch import ApplyPatch

import gzip
import pickle

import os

set_all_seed(0)

# Choose an integer in the range 0-9 to select the patch
patch_id = 3

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

# Build the preprocessing stack with ApplyPatch before the normalization step
preprocess = Compose([Resize(256), CenterCrop(224), ToTensor(),
                      apply_patch,
                      Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                      ])

# Load data
dataset = ImageFolder(os.path.join(os.getcwd(), "assets/data"),
                      transform=preprocess,
                      target_transform=target_transforms)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
x, y = next(iter(data_loader))  # load a mini-batch

imshow(torchvision.utils.make_grid(x.cpu().detach(), nrow=5), transforms=preprocess)
print("\nLabels:")
for y_i in y:
    print(target_to_classname[y_i.item()].split(',')[0])


# Load model
model = models.alexnet(pretrained=True)
model.eval()

# Test the patch on these data
output_adv = model(x)
predictions = torch.argmax(output_adv, dim=1).cpu().detach().numpy()
print("\nPredictions:")
for pred in predictions:
    print(target_to_classname[pred].split(',')[0])


