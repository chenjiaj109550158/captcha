# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import csv
import cv2

import random


from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torchvision import models
from matplotlib import pyplot as plt


class net_task1(nn.Module):
    def __init__(self):
        super(net_task1, self).__init__()
        
        self.resnet = models.resnet18(weights='DEFAULT')
        self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.d  = nn.Dropout(p=0.2)
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0) 
        # self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        # self.bn_1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        # self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)
        # self.bn_2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0) 
        # self.bn_3 = nn.BatchNorm2d(256)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(256, 512)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        
        # mean, std = x.mean([1,2]), x.std([1,2])
        # print(mean.shape)
        # exit()
        x = self.model_wo_fc(x)
        x = self.d(x)
        x = torch.flatten(x, 1)
        x = F.softmax(self.fc(x), dim=1)
        return x


def get_model(task_num):
    if task_num==1:
        model = net_task1()
    elif task_num ==2:
        model =  net_task1()
    elif task_num ==3:
        model = net_task1()
    else:
        print('invalid task num')
        exit()
    
    
    return model