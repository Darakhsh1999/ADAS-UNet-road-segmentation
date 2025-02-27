import torch
import torch.nn as nn
from data import RoadDataRuntimeLoad
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Resize, InterpolationMode
from pprint import pprint
from params import Params
from train import train, test, tune_threshold
from model import UNet
from utils import save_model

if __name__ == "__main__":

    # Parameters
    p = Params()

    # Model
    model = UNet()
    model.to(p.device)

    # Data splits
    transform = Resize((512,512), interpolation=InterpolationMode.NEAREST_EXACT)
    data = RoadDataRuntimeLoad(transform=transform)
    train_data = Subset(data, range(0,int(0.8*len(data))))
    val_data = Subset(data, range(int(0.8*len(data)),int(0.9*len(data))))
    test_data = Subset(data, range(int(0.9*len(data)), len(data)))
    assert (len(data) == (len(train_data)+len(val_data)+len(test_data))), "Discrepency in data subset lenghts"

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=p.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=p.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=p.batch_size, shuffle=False)

    # Optimizer and loss_fn 
    optimizer = torch.optim.SGD(model.parameters(), lr=p.lr, momentum=p.momentum)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=data.pos_weight)

    # Train model
    train(model, optimizer, loss_fn, p, train_loader, val_loader, test_loader)

    # Tune threshold
    best_theta = tune_threshold(p.n_tune, p.lower, p.upper, model, p, test_loader)
    print(f"Best theta v1 = {best_theta}")
    lower_v2 = max(0,best_theta-p.delta_tune)
    upper_v2 = min(0.99,best_theta+p.delta_tune)
    best_theta = tune_threshold(p.n_tune, lower_v2, upper_v2, model, p, test_loader)
    print(f"Best theta v2 = {best_theta}")

    # Test with optimal threshold
    test_performance_dict = test(model, p, test_loader, threshold=best_theta)
    pprint(test_performance_dict)

    # Save models
    model.to("cpu")
    save_model(model, save_path=p.save_path)

    print(f"Finished training. Model is saved in path: {p.save_path} | best tuned threshold = {best_theta}")