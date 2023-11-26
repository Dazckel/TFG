from SiameseDataset import *
from metrics import *
import numpy as np
from NeuralNetwork import *

#RUTAS
################################################################################################
# Establecemos las rutas a utilizar.
PATH_ROOT = Path(os.path.dirname(__file__)).parent  # Ruta de la carpeta padre
if operating_system == 'Windows':
    PATH_DATASET = Path('F:/') / 'Dataset/ADNI/NewImages'
elif operating_system == 'Linux':
    PATH_DATASET = PATH_ROOT / 'Datos/Dataset/ADNI/NewImages'  # Rutas hacia las imágenes
################################################################################################

# FUNCIÓN DE PÉRDIDA Y ACCURACY
loss_function = ContractiveLoss(0.5)
accuracy = torchmetrics.classification.BinaryAccuracy(validate_args=False)
accuracy = accuracy.cuda()
################################################################################################

# PARÁMETROS A PROBAR DURANTE EL ENTRENAMIENTO:
################################################################################################
MARGEN = 0.5
n_epochs = 8
unfreeze_epoch = 6
output = [100,125,150]
batch_size = [128]
modelos = ["resnet18","GoogleLenet"]
nn_output_size = [32,16,8,4]
input_lr = [0.02,0.005,0.001,0.0005]
ft_lr = 0.00001
create_dataset = False  # Indica si se crean de nuevo los datasets o no.
continuar = False  # Indica si se retoma un nuevo entrenamiento o si se inicia uno desde 0.
seed = 42
np.random.seed(seed)

if torch.cuda.device_count() == 1:
    device = 0
################################################################################################

# Load dataloader
################################################################################################
def getDataLoader(transformComp,bs):
    train_dataset = SiameseNetworkDataset(transform=transformComp,
                                          dataset_csv=os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV',
                                                                   'train_all_classes_T1_same.csv'))
    valid_dataset = SiameseNetworkDataset(transform=transformComp,
                                          dataset_csv=os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV',
                                                                   'train_all_classes_T1_same.csv'))
    test_dataset = SiameseNetworkDataset(transform=transformComp,
                                         dataset_csv=os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV',
                                                                  'train_all_classes_T1_same.csv'))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True)
    return train_loader, valid_loader, test_loader
################################################################################################