import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

from ML_Pipeline.network import UNetPP
from argparse import ArgumentParser
from albumentations.augmentations import transforms
from albumentations import Resize
from albumentations.core.composition import Compose


val_transform = Compose([
    Resize(256, 256),
    transforms.Normalize(),
])


def image_loader(image_name):
    # Your code goes here

