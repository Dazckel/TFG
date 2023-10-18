"""
@brief Este archivo implementa la red neuronal que se encargará de procesar los archivos
MRI para poder.

"""
import os
import random
from enum import Enum
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from itertools import combinations
import numpy as np
from sklearn.utils import resample
import nibabel as nb

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device.")

PATH_ROOT = Path(os.path.dirname(__file__)).parent
PATH_DATASET = PATH_ROOT / 'Datos/Dataset/ADNI/FINAL_ADNI'


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.cnn1 = nn.Sequential(
            nn.ReflecionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLu(inplace=True),
            nn.BatchNorm2d(p=.2),

            nn.Dropout2d(p=.2),
            nn.ReflecionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLu(inplace=True),
            nn.BatchNorm2d(p=.2),
            nn.Dropout2d(p=.2),

            nn.ReflecionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLu(inplace=True),
            nn.BatchNorm2d(p=.8),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLu(inplace=True),

            nn.Linear(500, 500),
            nn.ReLu(inplace=True),

            nn.Linear(500, 5)
        )

        def forward_once(self, x):
            output = self.cnn1(x)
            output = output.view(output.size()[0], -1)
            output = self.fc1(output)
            return output

        # def forward(self, input1, input2):
        #     output1 = self.forward_once(input1)
        #     output2 = self.forward_once(input2)
        #     return output1, output2

        def forward(self, input1, input2):
            output1 = self.resnet50(input1)
            output2 = self.resnet50(input2)
            return output1, output2


Diagnosis = Enum('Diagnosis', ["NL", "MCI", "AD"])
Diag = ["NL", "MCI", "AD"]


class ContractiveLoss(torch.nn.Module):
    """
        Contrastive loss function
    """

    def __init__(self, margin=2.0):
        super(ContractiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidian_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidian_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin) - euclidian_distance, min=0.0),
                                      2)
        return loss_contrastive


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, path_root=PATH_ROOT):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.number_NL = 0
        self.number_MCI = 0
        self.number_AD = 0

        self.NL = []
        self.MCI = []
        self.AD = []
        for img in os.listdir(self.imageFolderDataset):
            if Diag[0] in img:
                self.NL.append(img)
            elif Diag[1] in img:
                self.MCI.append(img)
            elif Diag[2] in img:
                self.AD.append(img)

        self.number_NL = len(self.NL)
        self.number_MCI = len(self.MCI)
        self.number_AD = len(self.AD)

        self.path_images = os.path.join(path_root, 'images')  # Ruta imágenes
        self.path_root = path_root  # Ruta archivos entrenamiento
        self.transform = transform  # Transformaciones

        # Si creamos un nuevo dataset.
        file_pairs, lbs = self.createSamples()

        self.file_names = file_pairs  # Pares de imágenes equitativamente distribuidos
        self.labels = lbs  # Etiquetas mostrando

    def createSamples(self):

        # Cogemos de todas las clases la misma cantidad de imágenes.
        # En este caso esa cantidad viene determinada por el número de imágenes
        # de la clase con menos imágenes.
        self.NL = self.NL[:self.number_AD]
        self.MCI = self.MCI[:self.number_AD]

        self.dataframe = pd.DataFrame({"NL": self.NL,
                                       "MCI": self.MCI,
                                       "AD": self.AD})

        filenames = []
        filenames.extend(self.NL)
        filenames.extend(self.MCI)
        filenames.extend(self.AD)

        classes = []
        for i in filenames:
            if i in self.NL:
                classes.append(Diag[0])
            elif i in self.MCI:
                classes.append(Diag[1])
            elif i in self.AD:
                classes.append(Diag[2])

        filenames_classes = dict(zip(filenames, classes))

        file_pairs = list(combinations(filenames, 2))

        file_pairs = list(set(file_pairs))

        # random_idxs = np.random.randint(0, len(file_pairs), size=len(file_pairs)).tolist()
        # file_pairs = [file_pairs[i] for i in random_idxs]
        lbs = []
        x0 = []
        x1 = []
        # Iteramos sobre los pares para conocer sus clases
        for filename1, filename2 in file_pairs:
            class1 = filenames_classes[filename1]
            class2 = filenames_classes[filename2]
            if class1 == class2:
                lbs.append(1)
            else:
                lbs.append(0)
            x0.append(filename1)
            x1.append(filename2)

        # Guardamos este dataset por si hemos de interrumpir el entrenamiento
        # y proseguir más adelante.
        new_df = pd.DataFrame(data=zip(x0, x1, lbs), columns=['x0', 'x1', 'Y'])

        df1 = new_df[new_df.Y == 0]
        df2 = new_df[new_df.Y == 1]
        samples = len(df2)
        df1_balanced = resample(df1, replace=True, n_samples=samples, random_state=2)
        df2_balanced = resample(df2, replace=True, n_samples=samples, random_state=2)

        dataframe = pd.concat([df1_balanced, df2_balanced]).reset_index()
        file_pairs = list(zip(dataframe.x0, dataframe.x1))

        lbs = list(dataframe.Y)

        return file_pairs, lbs

    def __getitem__(self, idx):
        img1 = self.file_names[idx][0]
        img2 = self.file_names[idx][1]

        img1 = nb.loadf(os.path.join(self.path_images, img1))
        img2 = nb.load(os.path.join(self.path_images, img2))
        label = self.labels[idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label, self.file_names[idx][0], self.file_names[idx][1]

    # Nos devuelce cuantas imágenes de cada clase tenemos.
    def getStatistic(self):
        total_img = len(os.listdir(self.imageFolderDataset))
        print(f'NL: {len(self.NL)}: {len(self.NL) / total_img * 100}%')
        print(f'MCI: {len(self.MCI)}: {len(self.MCI) / total_img * 100}%')
        print(f'AD: {len(self.AD)}: {len(self.AD) / total_img * 100}%')


# TRANSFORMACIONES SOBRE LOS DATOS
class Rescale(object):
    def __init__(self, output_size=80):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        resize = torchvision.transforms.Resize((self.output_size, self.output_size))
        new_img = resize(image)
        return new_img


class ToTensor(object):
    def __call__(self, image):
        return torch.from_numpy(image)


def save_to_csv(subset, path):
    print(path)
    labels = [None] * subset.__len__()
    imgs1 = [None] * subset.__len__()
    imgs2 = [None] * subset.__len__()
    for i, (_, _, label, img1, img2) in enumerate(subset):
        labels[i] = label
        imgs1[i] = img1
        imgs2[i] = img2

    new_df = pd.DataFrame(data=zip(imgs1, imgs2, labels), columns=["image1", "image2", "Y"])
    new_df.to_csv(path)


dataset = SiameseNetworkDataset(Path(os.path.dirname(__file__)).parent / 'Datos/Dataset/ADNI/FINAL_ADNI')
