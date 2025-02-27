import os
import numpy as np
import torch
import early_stopping

class Params:

    batch_size = 6
    threshold = 0.2
    lr = 0.001
    momentum = 0.9
    n_epochs = 60
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    save_path = os.path.join("models",f"unet{np.random.randint(0,1024)}.pt")
    optim_metric = "iou"
    stopping_criterion = early_stopping.EarlyStopping(patience=5)

    # Threshold tuning
    lower = 0.001
    upper = 0.9
    delta_tune = 0.2
    n_tune = 20
    