import os
import numpy as np
import torch

class Params:

    batch_size = 4
    threshold = 0.2
    lr = 0.001
    n_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    save_path = os.path.join("models",f"unet{np.random.randint(0,1024)}.pt")
    optim_metric = "iou"

    # Threshold tuning
    lower = 0.001
    upper = 0.3
    delta_tune = 0.2
    n_tune = 20
    