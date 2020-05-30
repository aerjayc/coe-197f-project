import torch
import os.path
from resnet import build_resnet

def lr_scheduler(epoch):
    lr = 1e-3
    if epoch > 80:
        lr *= 5e-2
    elif epoch > 60:
        lr *= 1e-1
    elif epoch > 40:
        lr *= 5e-1
    
    print('Learning rate:', lr)

    return lr

def save_model(model, weights_dir, weight_fname):
    weight_path = os.path.join(weights_dir, weight_fname)
    print('Saving weights to', weight_path)
    torch.save(model.state_dict(), weight_path)

def restore_weights(model, weights_dir, weight_fname):
    weight_path = os.path.join(weights_dir, weight_fname)
    print('Restoring weights from', weight_path)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
