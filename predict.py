import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from utils import load_model, numpy_to_input_tensor
from torchvision.transforms import Resize, InterpolationMode


def predict(model_path, source_path, threshold=0.3, FPS=30, use_camera=False):

    # Load in model
    device = "cuda" if torch.cuda.is_available else "cpu"
    model = load_model(model_path)
    model.to(device)

    # Load video source
    source = cv2.VideoCapture(0) if use_camera else cv2.VideoCapture(source_path)

    cv2.namedWindow("root", cv2.WINDOW_NORMAL)
    transform = Resize((512,512), interpolation=InterpolationMode.NEAREST_EXACT)
    deley = 1000//FPS

    with torch.no_grad():
        while cv2.waitKey(deley) != ord("q"):

            success, frame = source.read() # load in next frame

            if not success:
                print("failed frame")
                break

            # Predict
            torch_image = numpy_to_input_tensor(frame, transform) # frame [H,W,C] -> tensor [1,C,512,512]
            x = torch_image.to(device)
            x = model(x)
            x = F.sigmoid(x)
            x = 255*(x >= threshold).type(torch.uint8) # [1,1,512,512]
            
            x_np = x.cpu().numpy().squeeze() # [H,W]
            binary_mask = cv2.resize(x_np, (frame.shape[1],frame.shape[0]))
            binary_mask_stack = np.stack((binary_mask, binary_mask, binary_mask))
            binary_mask_stack = np.moveaxis(binary_mask_stack,0,-1)
            binary_mask_stack[:,:,1:3] = 0

            combined_frame = cv2.addWeighted(frame,1.0,binary_mask_stack,0.5,0)
            cv2.imshow("root", combined_frame)
        

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    model_path = os.path.join("models","unet996.pt")
    source_path = os.path.join("dataset","ScenicDrive_trim.mp4")
    predict(model_path, source_path, threshold=0.2)