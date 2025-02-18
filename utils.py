import torch
from model import UNet

def save_model(model:torch.nn.Module, save_path:str):
    torch.save(model.state_dict(), save_path)

def load_mode(model_file_path):
    model = UNet()
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    return model
