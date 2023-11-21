import torch
import torchvision.transforms as trfms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg', force=False)


def train(model, device, train_loader, optimizer, accuracy, criterion):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    train_loss = 0
    ratio = 0
    factor = 100
    toolbar_width = 50

    for batch_idx, (images_1, images_2, targets, _, _) in enumerate(train_loader):
        # Cargamos los datos del batch en la GPU, normalizamos y aplicamos transformaciones.
        images_1, images_2, targets = images_1.to(device).float(), images_2.to(device).float(), targets.to(
            device).float()

        # LAS IMÁGENES DE ADNI ESTÁN YA NORMALIZADAS APARENTEMENTE
        tr1 = trfms.Normalize(images_1.mean(), images_1.std())
        tr2 = trfms.Normalize(images_2.mean(), images_2.std())
        images_1 = tr1(images_1)
        images_2 = tr2(images_2)
        optimizer.zero_grad()
        #########################################################################################################
        output1, output2 = model(images_1, images_2)
        loss, outputs = criterion(output1, output2, targets)

        pred = torch.where(outputs > 1, 1, 0)
        accuracy.update(pred, targets)
        accuracy_batch = accuracy(pred, targets)


        # Aplicamos backprop
        try:
            loss.backward()
        except Exception as ex:
            return ex
        optimizer.step()

        if batch_idx % factor == 0 and batch_idx != 0:
            if batch_idx != 0:
                model.set_lastBatch(batch_idx)
                torch.save(model.state_dict(), model.get_path_model())
                torch.save(optimizer.state_dict(), model.get_path_optimizer())
                print("Model Saved")
                print(f'Last batch accuracy: {accuracy_batch.item() * 100}%')
            ratio = 0

        perc = batch_idx / len(train_loader)
        current_progress = int(np.ceil(perc * toolbar_width))
        print('#' * current_progress, '-' * (toolbar_width - current_progress), np.round(perc * 100, 2),
              '% de la época', end='\r')

        train_loss += loss.sum().item()
        #
        ratio += pred.eq(targets.view_as(pred)).sum().item()
