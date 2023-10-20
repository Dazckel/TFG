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
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)
        self.path_model = path_model
        self.path_optimizer = path_optimizer
        self.fc = nn.Sequential(
            nn.Linear(7680, 256),
            nn.Dropout(p=0.4, inplace=True),
            nn.SiLU(inplace=True),
            nn.Linear(256, 3),
        )
        self.lastBatch = lastBatch
        # Cargamos los pesos de la última tanda de entrenamiento
        if cont:
            if torch.cuda.device_count() == 1:
                self.load_state_dict(torch.load(self.path_model))
            else:
                self.load_state_dict(
                    torch.load(self.path_model, map_location=torch.device('cpu')))

    def get_path_model(self):
        return self.path_model

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

    def __init__(self, margin=2.0):
        super(ContractiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidian_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidian_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin) - euclidian_distance, min=0.0),
                                      2)
        return loss_contrastive
