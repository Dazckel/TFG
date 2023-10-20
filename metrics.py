import torch
import torchvision.transforms as trfms
import torchmetrics
import matplotlib.pyplot as plt


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
        pred = torch.where(outputs > 0.5, 0, 1)
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
