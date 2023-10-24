"""
@brief Este archivo implementa la red neuronal que se encargará de procesar los archivos
MRI para poder.

"""
import os
from enum import Enum
import torch
import torchvision
from torch import nn
from pathlib import Path

cont = False
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
    def __init__(self, path_model, path_optimizer, lastBatch=0):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=True)

        # Cambiamos la capa inicial de resnet para poder adaptarla a las imágenes MRI de un solo canal.
        # Por lo general las redes suelen aceptar imágenes a color, que tienen 3 canales.
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)
        self.path_model = path_model
        self.path_optimizer = path_optimizer

        # La última capa de nuestro modelo será una fullyconnected que devolverá las codificaciones
        # de las imágenes en vectores de 64 items.
        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.Dropout(p=0.4, inplace=True),
            nn.SiLU(inplace=True),
            nn.Linear(256, 32),
        )
        self.lastBatch = lastBatch
        # Cargamos los pesos de la última tanda de entrenamiento
        if cont:
            if torch.cuda.device_count() == 1:
                self.load_state_dict(torch.load(self.path_model))
            else:
                self.load_state_dict(
                    torch.load(self.path_model, map_location=torch.device('cpu')))
        else:
            self.resnet.apply(self.init_weights)
            self.fc.apply(self.init_weights)

    def get_path_model(self):
        return self.path_model

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_path_optimizer(self):
        return self.path_optimizer

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output1 = self.fc(output1)

        output2 = self.forward_once(input2)
        output2 = self.fc(output2)

        return output1, output2

    def set_lastBatch(self, idx):
        self.lastBatch = idx

    def get_lastBatch(self):
        return self.lastBatch


class ContractiveLoss(torch.nn.Module):
    """
        Contrastive loss function
    """

    def __init__(self, margin=0.75):
        super(ContractiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist_output = torch.sqrt(dist_sq)  # Distancia euclidea

        mdist = self.margin - dist_output
        dist = torch.clamp(mdist, min=0.0)
        loss = ((1 - y) * dist_sq + y * torch.pow(dist, 2)) / 2.0
        loss = torch.sum(loss) / x0.size()[0]
        return loss, dist_output
