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

if operating_system == 'Windows':
    PATH_DATASET = Path('F:/') / 'Dataset/ADNI/NewImages'
elif operating_system == 'Linux':
    PATH_DATASET = PATH_ROOT / 'Datos/Dataset/ADNI/Images'  # Rutas hacia las imágenes

if create_dataset:
    createDataset(PATH_DATASET)
n_epochs = 6
###############################################################################################################

# Creación de las transformaciones a aplicar.
# Escogemos como imagen de entrada a la red output

# Como función de pérdida usamod contractiveloss.
loss_function = ContractiveLoss()
accuracy = torchmetrics.classification.BinaryAccuracy(validate_args=False)
accuracy = accuracy.cuda()


# PARÁMETROS A PROBAR DURANTE EL ENTRENAMIENTO:

output = [100,125,150,200]
batch_size = [128]
modelos = ["resnet18","GoogleLenet"]
nn_output_size = [64, 32, 16, 8, 4]
input_lr = [0.01,0.005,0.001,0.0005]
ft_lr = 0.00001

for modelo in modelos:
    print(f"Entrenando modelo {modelo}.")
    for nn_os in nn_output_size:
        print(f"Codificación en {nn_os} valores.")
        for bs in batch_size:
            print(f"Batch size: {bs}.")
            for res in output:
                print(f"Resolución de salida: {res}")
                for init_lr in input_lr:
                    print(f"LR inicial: {init_lr}")
                    rescale = Rescale(output=res)
                    toTensor = ToTensor()
                    dir_model = Path(os.path.join(PATH_ROOT,f'Codigo/models/{modelo}_{nn_os}_{bs}_{res}_{init_lr}/'))
                    if not dir_model.exists():
                        os.mkdir(dir_model)
                    dir_optimizer = Path(os.path.join(PATH_ROOT,f'Codigo/optimizers/{modelo}_{nn_os}_{bs}_{res}_{init_lr}/'))
                    if not dir_optimizer.exists():
                        os.mkdir(dir_optimizer)

                    path_model = os.path.join(PATH_ROOT, f'Codigo/models/{modelo}_{nn_os}_{bs}_{res}/model_trained_.pt')  # Ruta de los pesos del modelo
                    path_optimizer = os.path.join(PATH_ROOT, f'Codigo/optimizers/{modelo}_{nn_os}_{bs}_{res}/optimizer_.pt')  # Ruta del optimizador.
                    transformComp = trfms.Compose([toTensor, rescale])  # Composición de transformaciones
                    ###############################################################################################################

                    # Modelo y optimizador
                    model = SiameseNetwork(path_model=path_model, path_optimizer=path_optimizer, lastBatch=0,modelo = modelo,output_size= nn_os).to(device)
                    # Como optimizador cogemos Adam con los parámetros por defecto.
                    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=[0.9, 0.99])

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

                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
                    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True)
                    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True)

                    final_train_accs = []
                    final_valid_accs = []
                    epoch_counter = 0
                    model.freeze()

                    for epoch in range(0, n_epochs):
                        print("Época :", epoch)
                        init = time.time()
                        ex = train(model, device, train_loader, optimizer, accuracy, loss_function)
                        if ex == RuntimeError:
                            break
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

                        if epoch_counter == 4:
                            model.unfreeze()
                            for g in optimizer.param_groups:
                                g['lr'] = ft_lr
                        epoch_counter += 1
                        print("=====================================================")

                    torch.save(model.state_dict(), model.get_path_model())
                    print("Final Model Saved")
                    showAccuracies(final_train_accs, final_valid_accs, range(0, n_epochs))

                    model.eval()
                    test_acc = accuracy_personal(model, test_loader, accuracy, device, loss_function)

