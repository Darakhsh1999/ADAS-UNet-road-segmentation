import torch
import torch.nn as nn

class Params:

    batch_size = 8
    lr = 0.001
    n_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    