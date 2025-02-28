import cv2
import numpy as np
import torch
from model import UNet

def numpy_to_input_tensor(numpy_frame, transform):
    """ numpy_frame [H,W,C] -> tensor [1,3,512,512] """
    numpy_frame = np.moveaxis(numpy_frame, -1, 0) # [C,H,W]
    torch_frame = torch.Tensor(numpy_frame[None]) # [1,C,H,W]
    torch_frame = transform(torch_frame)
    torch_frame = torch_frame.type(torch.float32) / 255.0 # [1,C,H,W]
    return torch_frame

def save_model(model:torch.nn.Module, save_path:str):
    torch.save(model.state_dict(), save_path)

def load_model(model_file_path):
    model = UNet()
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    return model