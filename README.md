# ADAS-UNet-road-segmentation

## Project Overview
Accurate road–background segmentation is a key prerequisite for Advanced Driver-Assistance Systems (ADAS), autonomous driving pipelines, and HD-map creation. This repository provides a lightweight yet effective U-Net-based approach that can run in real time on consumer-grade GPUs or even directly from a dash-cam feed using `use_camera=True`.

Key features:
* Custom annotation workflow via the [Road Segmentation Tool (RST)](https://github.com/Darakhsh1999/Road-Segmentation-Tool)
* Supports training, offline inference on images/videos, and live inference from a camera
* Small memory footprint — default input resolution of 512×512


## U-Net model that performs binary segmentation of roads

The dataset is annotated using my custom [Road Segmentation Tool (RST)](https://github.com/Darakhsh1999/Road-Segmentation-Tool) where contours constructed through continously connected lines and splines are interpolated as a function of time. The original size of the images are (720,1280) however the DataLoader downscales the images to (512,512) that are then passed through the U-Net. The model can also perform live inference using the output of a dashcam mounted on the front of a car by passing in the parameter <code>use_camera=True</code> in the <code>predict.py</code> module.

---

The model shows promising results and can in general differentiate between road and background pixels, however, due to the I/O bottleneck and low RAM caused by the dataset and limitation in hardware, I was not able to push for higher performance. Ideally one would require a significant increase in dataset size and a larger compute cluster to feasible be able to train a high-performing model. Despite this, my local model could achieve a validation intersection-over-union score of $IOU \approx 0.9$ .

## **Example**

Below is a recording shown of the model performing realtime inference on a video source and overlaying its binary road segmentation.

![Image](https://github.com/user-attachments/assets/a024c4b7-114e-40fe-9e1e-6035dd231288)

## **Data sample example**

Visual example of a some data samples

![Data sample 1](https://github.com/user-attachments/assets/3de4e8bf-15ab-492d-8a5b-d52e0d2c979e)

![Data sample 2](https://github.com/user-attachments/assets/907afe8a-ee7d-4cb1-a819-4d669ce762ab)
