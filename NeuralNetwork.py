"""
@brief Este archivo implementa la red neuronal que se encargará de procesar los archivos
MRI para poder.

"""
import random

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device.")


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
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        should_get_same_class = random.randint(0.1)
