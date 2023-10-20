from SiameseDataset import *
from metrics import *
from TrainLoop import train
import torchvision.transforms as trfms
import numpy as np
from NeuralNetwork import *
import time

create_dataset = False

cont = False
# Obtenemos la ruta del csv del dataset
PATH_ROOT = Path(os.path.dirname(__file__)).parent
PATH_DATASET = PATH_ROOT / 'Datos/Dataset/ADNI/FINAL_ADNI'

###############################################################################################################

# COMPOSICIÓN DE TRANSFORMACIONES A APLICAR
rescale = Rescale(output_size=75)
toTensor = ToTensor()
path_model = os.path.join(PATH_ROOT, 'models/model_trained.pt')
path_optimizer = os.path.join(PATH_ROOT, 'optimizers/optimizer.pt')
transformComp = trfms.Compose([toTensor, rescale])
###############################################################################################################

# Modelo y optimizador
model = SiameseNetwork(path_model=path_model, path_optimizer=path_optimizer, lastBatch=0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1 * (10 ** -4), betas=[0.9, 0.99])
criterion = ContractiveLoss()
accuracy = torchmetrics.classification.BinaryAccuracy(validate_args=False)
accuracy = accuracy.cuda()

if cont:
    if torch.cuda.device_count() == 1:
        model.load_state_dict(torch.load(path_model))
        optimizer.load_state_dict(torch.load(path_optimizer))
    else:
        optimizer.load_state_dict(
            torch.load(path_optimizer, map_location=torch.device('cpu')))
        model.load_state_dict(
            torch.load(path_model, map_location=torch.device('cpu')))
if create_dataset:
    createDataset(PATH_DATASET)

###############################################################################################################

if torch.cuda.device_count() == 1:
    device = 0

###############################################################################################################

# Datasets y dataloader
seed = 42
batch_size = 8
np.random.seed(seed)

# Cargamos los datasets que previamente hemos guardado en los csvs
train_dataset = SiameseNetworkDataset(transform=transformComp,
                                      dataset_csv=os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV',
                                                               'train_all_classes.csv'))
# valid_dataset = SiameseNetworkDataset(transform=transformComp,
#                                       dataset_csv=os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV',
#                                                                'valid_all_classes.csv'))
# test_dataset = SiameseNetworkDataset(transform=transformComp,
#                                      dataset_csv=os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV',
#                                                               'test_all_classes.csv'))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

final_train_accs = []
final_valid_accs = []

for epoch in range(1, 9):
    print("Época :", epoch)
    init = time.time()
    train(model, device, train_loader, optimizer, train_dataset, accuracy, criterion)
    end = time.time()
    print("\tTraining time: ", end - init)

    print("Evaluando en train...")
    init = time.time()
    final_train_accs.append(accuracy.compute().item())
    accuracy.reset()
    end = time.time()
    print("\tTraining evaluation time: ", end - init)
    print("Evaluando en validación...")
    init = time.time()
    valid_acc = accuracy_personal(model, valid_loader, accuracy, device, criterion)
    end = time.time()
    print("\tValidation evaluation time: ", end - init)
    print("\tValidation accuracy: ", valid_acc)
    final_valid_accs.append(valid_acc)

    print("=====================================================")

torch.save(model.state_dict(), model.get_path_model())
print("Final Model Saved")
