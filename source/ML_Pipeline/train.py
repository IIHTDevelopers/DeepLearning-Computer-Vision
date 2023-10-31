import os
import yaml
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from ML_Pipeline.utils import AverageMeter,iou_score
from albumentations import Resize
from albumentations.augmentations import transforms
from sklearn.model_selection import train_test_split
from albumentations.core.composition import Compose, OneOf
#from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.transforms import RandomRotate90
from ML_Pipeline.network import UNetPP
from ML_Pipeline.dataset import DataSet


def train(deep_sup, train_loader, model, criterion, optimizer):
    # Your code goes here