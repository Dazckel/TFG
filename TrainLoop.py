from trainloop_utils import *
from SiameseDataset import norm_st
def train(model, device, train_loader, optimizer, accuracy, criterion,epoch,bs,res):

    #vars
    train_loss = []

    torch.autograd.set_detect_anomaly(True)
    model.train()

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):

        # To GPU
        #########################################################################################################
        images_1, images_2, targets = images_1.to(device).float(), images_2.to(device).float(), targets.to(
            device).float()

        #########################################################################################################
        # LAS IMÁGENES DE ADNI ESTÁN YA NORMALIZADAS APARENTEMENTE
        #########################################################################################################
        images_1 = norm_st(images_1,res,bs)
        images_2 = norm_st(images_2,res,bs)
        #########################################################################################################

        #Predict
        output1, output2 = model(images_1, images_2)

        loss, outputs = criterion(output1, output2, targets)
        pred = torch.where(outputs > 0.5, 1, 0)
        accuracy.update(pred, targets)
        print(loss)

        # Aplicamos backprop
        #########################################################################################################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #########################################################################################################

        saveModel(batch_idx, model, optimizer,epoch)
        statusBar(batch_idx, train_loader)

        train_loss.append(loss.sum().item())


    return np.array(train_loss).mean()