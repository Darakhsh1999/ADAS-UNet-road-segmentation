import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_video, read_video_timestamps
from torchvision.transforms import Resize, InterpolationMode
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

import warnings
warnings.filterwarnings('ignore', module='torchvision')


def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

class RoadData(Dataset):

    def __init__(self, end_pts=50, transform=None, verbose=False):
        self.video_path = os.path.join("dataset","ScenicDrive_trim.mp4")
        self.binary_video_path = os.path.join("dataset","labeled_video1578.avi")
        self.transform = transform
        self.verbose = verbose

        # Load in videos Tensor[T,H,W,C] uint8
        if self.verbose: print("Reading in videos to memory...")
        self.mask, _, self.mask_config = read_video(self.binary_video_path, pts_unit="sec", end_pts=end_pts, output_format="TCHW") 
        self.mask = self.mask[:,[0],:,:] # extract single channel
        self.video, _, self.video_config = read_video(self.video_path, pts_unit="sec", end_pts=end_pts, output_format="TCHW")
        if self.verbose: print("Finished reading in videos")
        if self.verbose: print(f"Mask bytes {self.mask.nbytes/float(1073741824):.3f} GB")
        if self.verbose: print(f"Video bytes {self.video.nbytes/float(1073741824):.3f} GB")
        assert len(self.mask) == len(self.video), f"Mask frames: {len(self.mask)} differs from video frames: {len(self.video)}"
        self.n_frames = len(self.mask)

        # Calculate pos_weight
        n_positive = torch.count_nonzero(self.mask)
        self.pos_weight = (self.mask.nelement() - n_positive) / n_positive # pos_weight = negative/positive


    def __getitem__(self, idx):
        sample = self.video[idx], self.mask[idx] # ([C,H,W],[1,H,W])
        if self.transform:
            img, mask = sample
            sample = (self.transform(img), (255*(self.transform(mask) > 200).type(torch.uint8)))
        return sample

    def __len__(self):
        return self.n_frames

    def get_data_pair(self, n=5):
        
        sample_idx = np.random.choice(self.n_frames, n, replace=False)
        sample_list = []
        for idx in sample_idx:
            sample_list.append(self.__getitem__(idx))
        return sample_list
    
    def plot_example(self, n=3):
        sample_list = self.get_data_pair(n=n)
        for frame, mask in sample_list:
            frame, mask = np.moveaxis(frame.numpy(),0,-1), mask.numpy().squeeze()

            # Plotting
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(frame)
            axs[0].set_title("Image")
            axs[1].imshow(mask, cmap="gray")
            axs[1].set_title("Mask")
            plt.show()
    

class RoadDataRuntimeLoad(Dataset):

    def __init__(self, transform=None, verbose=0):
        self.video_path = os.path.join("dataset","ScenicDrive_trim.mp4")
        self.binary_video_path = os.path.join("dataset","labeled_video1578.avi")
        self.transform = transform
        self.verbose = verbose

        self.video_timestamps = read_video_timestamps(self.video_path)[0]
        self.mask_timestamps = read_video_timestamps(self.binary_video_path)[0]
        if self.verbose > 0:
            print(f"n_frames_video = {len(self.video_timestamps)}")
            print(f"n_frames_mask = {len(self.mask_timestamps)}")


        # Cropping frames
        _minlen = min(len(self.video_timestamps), len(self.mask_timestamps))
        self.video_timestamps = self.video_timestamps[:_minlen]
        self.mask_timestamps = self.mask_timestamps[:_minlen]
        assert len(self.video_timestamps) == len(self.mask_timestamps), f"N_frames_video = {len(self.video_timestamps)}, N_frames_mask = {len(self.mask_timestamps)}"
        self.n_frames = len(self.video_timestamps)

        # Estimate pos_weight
        self.estimate_pos_weight()


    def __getitem__(self, idx):
        video_idx = self.video_timestamps[idx]
        mask_idx = self.mask_timestamps[idx]
        video_frame = read_video(self.video_path, start_pts=video_idx, end_pts=video_idx, output_format="TCHW")[0][0] # [C,H,W]
        mask_frame = read_video(self.binary_video_path, start_pts=mask_idx, end_pts=mask_idx, output_format="TCHW")[0][0,[0],:,:] # [1,H,W]

        if self.transform:
            return (self.transform(video_frame), (255*(self.transform(mask_frame) > 100).type(torch.uint8)))
        return (video_frame, mask_frame) # [C,H,W]

    def __len__(self):
        return self.n_frames

    def estimate_pos_weight(self):
        _mask = read_video(self.binary_video_path, end_pts=self.mask_timestamps[int(0.1*len(self.mask_timestamps))])[0] # [T,H,W,C]
        _mask = _mask[:,:,:,0] # [T,H,W]
        assert _mask.ndim == 3, f"mask has shape: {_mask.shape}"
        n_positive = torch.count_nonzero(_mask)
        self.pos_weight = (_mask.nelement() - n_positive) / n_positive # pos_weight = negative/positive

    def get_data_pair(self, n=5):
        
        sample_idx = np.random.choice(self.n_frames, n, replace=False)
        sample_list = []
        for idx in sample_idx:
            sample_list.append(self.__getitem__(idx))
        return sample_list
    
    def plot_example(self, n=3, plot_type="subplot"):
        sample_list = self.get_data_pair(n=n)

        for frame, mask in sample_list:
            frame = np.moveaxis(frame.numpy().squeeze(),0,-1) # [H,W,C]
            mask = mask.numpy().squeeze() # [H,W]
            if plot_type == "subplot":

                    # Plotting
                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(frame)
                    axs[0].set_title("Image")
                    axs[1].imshow(mask, cmap="gray")
                    axs[1].set_title("Mask")
                    plt.show()
            elif plot_type == "overlay":
                    stacked_mask = np.stack((mask,mask,mask), axis=-1)
                    stacked_mask[:,:,0:2] = 0
                    combined_frame = cv2.addWeighted(frame,1.0,stacked_mask,0.3,0)
                    plt.imshow(combined_frame)
                    plt.show()
            else:
                ValueError(f"Unknown plot type: {plot_type}")


    

if __name__ == "__main__":

    transform = Resize((512,512), interpolation=InterpolationMode.NEAREST_EXACT)
    data = RoadDataRuntimeLoad(transform=transform)

    train_data = Subset(data, range(0,int(0.5*len(data))))
    val_data = Subset(data, range(int(0.5*len(data)),len(data)))

    train_data.dataset.plot_example(n=10, plot_type="overlay")
    val_data.dataset.plot_example(n=10, plot_type="subplot")

