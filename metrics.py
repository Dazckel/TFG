import torch
import torchvision.transforms as trfms
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import pandas as pd

def accuracy_personal(model, loader, accuracy: torchmetrics.Accuracy, device, criterion):
    accuracies = []
    images = []
    for images_1, images_2, targets, _, _ in loader:
        images_1, images_2, targets = images_1.to(device).float(), images_2.to(device).float(), targets.to(
            device).float()
        tr1 = trfms.Normalize(images_1.mean(), images_1.std())
        tr2 = trfms.Normalize(images_2.mean(), images_2.std())
        images_1 = tr1(images_1)
        images_2 = tr2(images_2)
        output1, output2 = model(images_1, images_2)
        loss, outputs = criterion(output1, output2, targets)
        pred = torch.where(outputs > 1, 1, 0)
        # if enter:
        #     print("Preds en valid: ", pred)
        #     enter = False
        batch_acc = accuracy(pred, targets)
        accuracies.append(batch_acc.item() * outputs.shape[0])
        images.append(outputs.shape[0])

    return sum(accuracies) / sum(images)


def showAccuracies(train_accs, valid_accs, iterations):
    plt.title("Accuracies on each epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(iterations, train_accs)
    plt.plot(iterations, valid_accs)
    plt.legend(["Training Accuracy", "Valid Accuracy"])
    plt.show()





def print_confusion_mat(model, loader, criterion, device):
    y_pred = []
    y_true = []
    toolbar_width = 50
    # ***TBD***
    # iterate over test data
    for batch_idx, (images_1, images_2, targets, _, _) in enumerate(loader):
        images_1, images_2, targets = images_1.to(device).float(), images_2.to(device).float(), targets.to(
            device).float()
        tr1 = trfms.Normalize(images_1.mean(), images_1.std())
        tr2 = trfms.Normalize(images_2.mean(), images_2.std())
        images_1 = tr1(images_1)
        images_2 = tr2(images_2)
        output1, output2 = model(images_1, images_2)
        loss, outputs = criterion(output1, output2, targets)
        pred = torch.where(outputs > 1, 1, 0)

        y_true.extend(targets.cpu().numpy().astype(np.uint8))
        y_pred.extend(pred.cpu().numpy().astype(np.uint8))

        perc = batch_idx / len(loader)
        current_progress = int(np.ceil(perc * toolbar_width))
        print('#' * current_progress, '-' * (toolbar_width - current_progress), np.round(perc * 100, 2),
              '% del dataloader', end='\r')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix)
    sn.heatmap(df_cm, annot=True)