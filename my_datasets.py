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



class Task1Dataset(Dataset):
    def __init__(self, data, root, transforms, return_filename=False):
        self.data = [sample for sample in data if sample[0].startswith("task1")]
        self.return_filename = return_filename
        self.root = root
        self.transforms = transforms
    
    def __getitem__(self, index):
        filename, label = self.data[index]
        img = Image.open(f"{self.root}/{filename}")
        img = self.transforms(img)
        # img = cv2.resize(img, (32, 32))
        # img = np.mean(img, axis=2)
        img = np.array(img)

        if self.return_filename:
            return torch.FloatTensor(img), filename
        else:
            return torch.FloatTensor(img), int(label)

    def __len__(self):
        return len(self.data)


def get_dataset(task_num, data, root, transforms, return_filename=False):
    if task_num==1:
        model = Task1Dataset(data, root, transforms, return_filename)
    elif task_num ==2:
        model =  Task1Dataset()
    elif task_num ==3:
        model = Task1Dataset()
    else:
        print('invalid task num')
        exit()
    
    
    return model