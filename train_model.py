import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from params import Params
from train import train
from model import UNet
from utils import save_model

# Parameters
p = Params()

# Model
model = UNet()

# Data sets
train_loader = DataLoader(train_data, batch_size=p.batch_size, shuffle=True)
val_loader = DataLoader(train_data, batch_size=p.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=p.batch_size, shuffle=False)

# Optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)

train(model, optimizer, p, train_loader, val_loader)

# Save models
model.to("cpu")
save_model(model)