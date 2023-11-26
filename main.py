from utils import *
from TrainLoop import train
import pandas as pd
import torchvision.transforms as trfms
if create_dataset:
    createDataset(PATH_DATASET)

###############################################################################################################
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

                    #Create path for model and optimizer
                    ###############################################################################################################
                    dir_model = Path(os.path.join(PATH_ROOT,f'Codigo\\models\\{modelo}_{nn_os}_{bs}_{res}_{init_lr}\\'))
                    dir_optimizer = Path(os.path.join(PATH_ROOT, f'Codigo\\optimizers\\{modelo}_{nn_os}_{bs}_{res}_{init_lr}\\'))
                    if not dir_model.exists():
                        os.mkdir(dir_model)
                    if not dir_optimizer.exists():
                        os.mkdir(dir_optimizer)
                    path_model = os.path.join(PATH_ROOT, dir_model,f'model_trained_.pt')  # Ruta de los pesos del modelo
                    path_optimizer = os.path.join(PATH_ROOT, dir_model,f'optimizer_.pt')  # Ruta del optimizador.
                    ###############################################################################################################

                    #Transform
                    ###############################################################################################################
                    rescale = Rescale(output=res)
                    toTensor = ToTensor()
                    transformComp = trfms.Compose([toTensor, rescale])  # Composición de transformaciones
                    ###############################################################################################################

                    # Modelo y optimizador
                    ###############################################################################################################
                    model = SiameseNetwork(path_model=path_model, path_optimizer=path_optimizer, lastBatch=0,modelo = modelo,output_size= nn_os).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=[0.9, 0.99])
                    ###############################################################################################################


                    # Cargamos los datasets que previamente hemos guardado en los csvs
                    ###############################################################################################################
                    train_loader, valid_loader, test_loader = getDataLoader(transformComp,bs)
                    ###############################################################################################################
                    final_train_accs = []
                    final_valid_accs = []
                    final_train_loss = []
                    final_valid_loss = []
                    ###############################################################################################################

                    ####################################################################################################
                    epoch_counter = 0
                    model.freeze()
                    for epoch in range(0, n_epochs):
                        print("Época :", epoch)

                        train_loss = train(model, device, train_loader, optimizer, accuracy, loss_function,epoch,bs,res)

                        final_train_loss.append(train_loss)
                        final_train_accs.append(accuracy.compute().item())
                        accuracy.reset()

                        ################################################################################################
                        valid_acc, loss_acc = accuracy_personal(model, valid_loader, accuracy, device, loss_function)
                        print("\tValidation accuracy: ", valid_acc)
                        ################################################################################################

                        #Refgister stats.
                        ################################################################################################
                        final_valid_accs.append(valid_acc)
                        final_valid_loss.append(loss_acc)
                        ################################################################################################

                        #Unfreeze
                        ################################################################################################
                        if epoch_counter == unfreeze_epoch:
                            model.unfreeze(optimizer,ft_lr)
                        ################################################################################################

                        epoch_counter += 1
                        print("=====================================================")

                    torch.save(model.state_dict(), model.get_path_model()[:-3]+f"{epoch}.pt")

                    stats = pd.Dataframe({"TrainLoss":final_train_loss,
                                  "TrainAcc": final_train_accs,
                                  "ValidLoss": final_valid_loss,
                                  "ValidAcc": final_valid_accs})
                    stats.to_csv(dir_model.__str__() + "\\stats.csv")



