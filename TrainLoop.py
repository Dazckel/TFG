import torch
import torchvision.transforms as trfms
import numpy as np


def train(model, device, train_loader, optimizer, dataset, accuracy, criterion):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    train_loss = 0
    correct = 0
    ratio = 0
    factor = 100

    toolbar_width = 50

    for batch_idx, (images_1, images_2, targets, _, _) in enumerate(train_loader):
        # Cargamos los datos del batch en la GPU, normalizamos y aplicamos transformaciones.
        images_1, images_2, targets = images_1.to(device).float(), images_2.to(device).float(), targets.to(
            device).float()
        tr1 = trfms.Normalize(images_1.mean(), images_1.std())
        tr2 = trfms.Normalize(images_2.mean(), images_2.std())
        images_1 = tr1(images_1)
        images_2 = tr2(images_2)
        optimizer.zero_grad()
        #########################################################################################################

        # Predicción
        output1, output2 = model(images_1[:, None, :, :], images_2[:, None, :, :])
        # Aplicamos función de pérdida
        loss, outputs = criterion(output1, output2, targets)

        # Comprobaciones sobre el rendimiento en este batch
        pred = torch.where(outputs > 0.5, 0, 1)
        accuracy.update(pred, targets)

        # Aplicamos backprop
        loss.backward()
        optimizer.step()

        if batch_idx % factor == 0 and batch_idx != 0:
            # # Cada factor batches, guardaos el modelo y las combinaciones procesadas.
            #   Cada vez que se realizan batch_size*factor predicciones guardamos el modelo y el optimizer
            # y ajustamos el learning rate
            if batch_idx != 0:
                model.set_lastBatch(batch_idx)
                torch.save(model.state_dict(), model.get_path_model())
                torch.save(optimizer.state_dict(), model.get_path_optimizer())
                print("Model Saved")
            ratio = 0

        perc = batch_idx / len(train_loader)
        current_progress = int(np.ceil(perc * toolbar_width))
        print('#' * current_progress, '-' * (toolbar_width - current_progress), np.round(perc * 100, 2),
              '% de la época', end='\r')

        train_loss += loss.sum().item()
        #
        ratio += pred.eq(targets.view_as(pred)).sum().item()
