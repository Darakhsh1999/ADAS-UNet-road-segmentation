import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import Resize, InterpolationMode
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

class RoadData(Dataset):

    def __init__(self, end_pts=50, transform=None, verbose=False):
        self.video_path = os.path.join("dataset","ScenicDrive_trim.mp4")
        self.binary_video_path = os.path.join("dataset","labeled_video7465.avi")
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
    
    def density_based_weighting(self):
        pass


class RoadDataLazyLoad(Dataset):
    pass

if __name__ == "__main__":

    transform = Resize((512,512), interpolation=InterpolationMode.NEAREST_EXACT)
    data = RoadData(end_pts=50, transform=transform)
    data.plot_example(4)