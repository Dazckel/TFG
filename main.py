from SiameseDataset import *
from metrics import *
from TrainLoop import train
import torchvision.transforms as trfms
import numpy as np
from NeuralNetwork import *
import time

create_dataset = False  # Indica si se crean de nuevo los datasets o no.
continuar = False  # Indica si se retoma un nuevo entrenamiento o si se inicia uno desde 0.

# Establecemos las rutas a utilizar.
PATH_ROOT = Path(os.path.dirname(__file__)).parent  # Ruta de la carpeta padre
PATH_DATASET = PATH_ROOT / 'Datos/Dataset/ADNI/FINAL_ADNI'  # Rutas hacia las imágenes

n_epochs = 10;
###############################################################################################################

# Creación de las transformaciones a aplicar.
# Escogemos como imagen de entrada a la red output
output = 150
batch_size = 16
rescale = Rescale(output=output)
toTensor = ToTensor()
path_model = os.path.join(PATH_ROOT, 'Codigo/models/model_trained.pt')  # Ruta de los pesos del modelo
path_optimizer = os.path.join(PATH_ROOT, 'Codigo/optimizers/optimizer.pt')  # Ruta del optimizador.
transformComp = trfms.Compose([toTensor, rescale])  # Composición de transformaciones
###############################################################################################################

# Modelo y optimizador
model = SiameseNetwork(path_model=path_model, path_optimizer=path_optimizer, lastBatch=0).to(device)
# Como optimizador cogemos Adam con los parámetros por defecto.
optimizer = torch.optim.Adam(model.parameters(), lr=1 * (10 ** -4), betas=[0.9, 0.99])
# Como función de pérdida usamod contractiveloss.
loss_function = ContractiveLoss(0.5)
accuracy = torchmetrics.classification.BinaryAccuracy(validate_args=False)
accuracy = accuracy.cuda()

if continuar:
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
np.random.seed(seed)

# Cargamos los datasets que previamente hemos guardado en los csvs
train_dataset = SiameseNetworkDataset(transform=transformComp,
                                      dataset_csv=os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV',
                                                               'train_all_classes.csv'))
valid_dataset = SiameseNetworkDataset(transform=transformComp,
                                      dataset_csv=os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV',
                                                               'valid_all_classes.csv'))
test_dataset = SiameseNetworkDataset(transform=transformComp,
                                     dataset_csv=os.path.join(PATH_ROOT, 'Codigo', 'DatasetsCSV',
                                                              'test_all_classes.csv'))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

final_train_accs = []
final_valid_accs = []

for epoch in range(0, n_epochs):
    print("Época :", epoch)
    init = time.time()
    train(model, device, train_loader, optimizer, train_dataset, accuracy, loss_function)
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
    valid_acc = accuracy_personal(model, valid_loader, accuracy, device, loss_function)
    end = time.time()
    print("\tValidation evaluation time: ", end - init)
    print("\tValidation accuracy: ", valid_acc)
    final_valid_accs.append(valid_acc)

    print("=====================================================")

torch.save(model.state_dict(), model.get_path_model())
print("Final Model Saved")

showAccuracies(final_train_accs, final_valid_accs, range(0, n_epochs))
