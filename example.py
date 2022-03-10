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


"""
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, \
    Normalize
from torchvision.models import alexnet
import torch

from utils.visualization import imshow
from utils.utils import set_all_seed

from transforms.apply_patch import ApplyPatch

import gzip
import pickle

import os

set_all_seed(0)

# classes of the small chunk of data
imagenette_classes = [0, 217, 482, 491, 497]

# dictionary with the label names
with open(os.path.join(os.getcwd(), "assets/imagenet1000_clsidx_to_labels.txt")) as f:
    target_to_classname = eval(f.read())

# Load the patches
with gzip.open(os.path.join(os.getcwd(), "assets/patches.gz"), 'rb') as f:
    patches, targets = pickle.load(f)

patch_id = 1

# Instantiate the ApplyPatch layer, setting patch, target and transforms
apply_patch = ApplyPatch(patches[patch_id], targets[patch_id],
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

# Load data
dataset = ImageFolder(os.path.join(os.getcwd(), "assets/data"), preprocess)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
x, y = next(iter(data_loader))
for i, y_i in enumerate(y):
    y[i] = imagenette_classes[y[i]]

imshow(torchvision.utils.make_grid(x.cpu().detach()), transforms=preprocess)


# Load model
model = alexnet(pretrained=True)

# Test the patch on these data
n_test_samples = 0
acc_clean = 0
with torch.no_grad():
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    n_test_samples += x.shape[0]

    # compute clean predictions
    output_clean = model(x)
    acc_clean += torch.sum(
        torch.argmax(output_clean, dim=1) == y).cpu().detach().numpy()
acc_clean /= n_test_samples
print(acc_clean)
print("")

