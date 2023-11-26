"""
@brief Este archivo implementa la red neuronal que se encargará de procesar los archivos
MRI para poder.

"""
import os
import torch
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision.models import GoogLeNet_Weights
from torchvision.models.alexnet import AlexNet_Weights
from torch import nn
from pathlib import Path
import math

def Log2(x):
    return (math.log10(x) /
            math.log10(2))

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
    def __init__(self, path_model, path_optimizer, lastBatch=0,modelo = "resnet18",output_size = 8):

        super(SiameseNetwork, self).__init__()
        self.name = modelo
        ################################################################################################################
        if self.name == "resnet18":
            self.model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)
            self.fc_input = 512
            self.model.fc = getfc(self.fc_input,output_size)
        elif self.name == "GoogleLenet":
            self.model = torchvision.models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
            self.fc_input = 1024
            self.model.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False),
                nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))
            self.model.fc = getfc(self.fc_input, output_size)
        elif self.name == "AlexNet":
            self.model = torchvision.models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        ################################################################################################################

        ################################################################################################################
        self.path_model = path_model
        self.path_optimizer = path_optimizer
        ################################################################################################################

        self.lastBatch = lastBatch
    def get_path_model(self):
        return self.path_model

    def freeze(self):

        if not self.name == "AlexNet":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.conv1.parameters():
                param.requires_grad = True
            for param in self.model.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.features[0].parameters():
                param.requires_grad = True


    def unfreeze(self,optimizer,ft_lr):
        for param in self.model.parameters():
            param.requires_grad = True

        for g in optimizer.param_groups:
            g['lr'] = ft_lr

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_path_optimizer(self):
        return self.path_optimizer

    def forward_once(self, x):
        output = self.model(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)

        output2 = self.forward_once(input2)

        return output1, output2

    def set_lastBatch(self, idx):
        self.lastBatch = idx

    def get_lastBatch(self):
        return self.lastBatch


class ContractiveLoss(torch.nn.Module):
    """
        Contrastive loss function
    """

    def __init__(self, margin=0.5):
        super(ContractiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist_output = torch.sqrt(dist_sq)  # Distancia euclidea
        mdist = self.margin - dist_output
        dist = torch.clamp(mdist, min=0.0)
        loss = ((1 - y) * dist_sq + y * torch.pow(dist, 2))
        loss = torch.mean(loss)
        # ContrastiveLoss()
        return loss, dist_output



def getfc(in_size, output_size):
    n_layers = math.ceil(Log2(in_size)) - math.ceil(Log2(output_size))
    seq = nn.Sequential()
    for i in range(0, n_layers):
        in_ = in_size / math.pow(2, i)
        out_ = in_size / math.pow(2, i + 1)
        seq.append(nn.Linear(in_features=int(in_), out_features=int(out_)))
        if i < n_layers-1:
            seq.append(nn.ReLU(inplace=True))

    return seq