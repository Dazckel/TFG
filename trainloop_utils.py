import torch
import numpy as np

factor = 100
toolbar_width = 50
def saveModel(batch_idx,model,optimizer,epoch):
    if batch_idx % factor == 0 and batch_idx != 0:
        if batch_idx != 0:
            model.set_lastBatch(batch_idx)
            torch.save(model.state_dict(), model.get_path_model()[:-3] + f'{epoch}.pt')
            torch.save(optimizer.state_dict(), model.get_path_optimizer()[:-3] + f'{epoch}.pt')

def statusBar(batch_idx,train_loader):
    perc = batch_idx / len(train_loader)
    current_progress = int(np.ceil(perc * toolbar_width))
    print('#' * current_progress, '-' * (toolbar_width - current_progress), np.round(perc * 100, 2),
          '% de la época', end='\r')