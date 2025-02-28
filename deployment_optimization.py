""" TensorRT optimization of deployed model """
# https://github.com/pytorch/TensorRT
# https://pytorch.org/TensorRT/
# https://pytorch.org/TensorRT/user_guide/using_dla.html

import os
import torch
import time
import torch_tensorrt
from utils import load_model

def compile_to_tensorrt(model, input_example):
    """ JIT tensorRT compilation """
    optimized_model = torch.compile(model, backend="tensorrt")
    optimized_model(input_example) # compiled on first run
    return optimized_model

def export_to_tensorrt(model, model_path, input_example, compile_specs={}):
    """ AOT tensorRT compilation and export """
    trt_gm = torch_tensorrt.compile(model, kwarg_inputs=compile_specs, ir="dynamo", inputs=input_example)
    torch_tensorrt.save(trt_gm, model_path[:-2]+"ep", inputs=input_example) 

def load_exported_tensorrt(compiled_model_path):
    return torch.export.load(compiled_model_path).module()

# Load model
model_path = os.path.join("models","unet996.pt")
model = load_model(model_path)
model.cuda()

# Define example input
input_image = torch.randn((1, 3, 512, 512)).cuda() # define what the inputs to the model will look like


compiled_model = compile_to_tensorrt(model, input_image)


# BENCHMARKS
N = 100

# Normal model
t = time.time()
for i in range(N):
    a = model(input_image)
t = time.time() - t
print(f"N={N} inference calls for normal model took t={t:.4f} s")

# Compiled model
t = time.time()
for i in range(N):
    a = compiled_model(input_image)
t = time.time() - t
print(f"N={N} inference calls for tensorrt model took t={t:.4f} s")