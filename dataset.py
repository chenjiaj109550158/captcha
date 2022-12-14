# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:3]:
        print(os.path.join(dirname, filename))
    if len(filenames) > 3:
        print("...")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import csv
import cv2
import numpy as np
import random
import os

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

TRAIN_PATH = "E:/ml/hw5/train"
TEST_PATH = "/kaggle/input/captcha-hacker/test"
device = "cpu"
# try device = "cuda" 
# and change your settings/accelerator to GPU if you want it to run faster


class Task1Dataset(Dataset):
    def __init__(self, data, root, transforms, return_filename=False):
        self.data = [sample for sample in data if sample[0].startswith("task1")]
        self.return_filename = return_filename
        self.root = root
    
    def __getitem__(self, index):
        filename, label = self.data[index]
        img = cv2.imread(f"{self.root}/{filename}")
        img = cv2.resize(img, (32, 32))
        img = np.mean(img, axis=2)
        if self.return_filename:
            return torch.FloatTensor((img - 128) / 128), filename
        else:
            return torch.FloatTensor((img - 128) / 128), int(label)

    def __len__(self):
        return len(self.data)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 10)
        )
        
        
    def forward(self, x):
        b, h, w = x.shape
        x = x.view(b, h*w)
        return self.layers(x)


train_data = []
val_data = []

with open(f'{TRAIN_PATH}/annotations.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        if random.random() < 0.7:
            train_data.append(row)
        else:
            val_data.append(row)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
train_ds = Task1Dataset(train_data, root=TRAIN_PATH, transforms=transforms)
# print(train_ds[0])
# exit()


train_dl = DataLoader(train_ds, batch_size=500, num_workers=4, drop_last=True, shuffle=True)

val_ds = Task1Dataset(val_data, root=TRAIN_PATH)
val_dl = DataLoader(val_ds, batch_size=500, num_workers=4, drop_last=False, shuffle=False)


model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


for epoch in range(100):
    print(f"Epoch [{epoch}]")
    model.train()
    for image, label in train_dl:
        image = image.to(device)
        label = label.to(device)
        
        pred = model(image)
        loss = loss_fn(pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    sample_count = 0
    correct_count = 0
    model.eval()
    for image, label in val_dl:
        image = image.to(device)
        label = label.to(device)
        
        pred = model(image)
        loss = loss_fn(pred, label)
        
        pred = torch.argmax(pred, dim=1)
        
        sample_count += len(image)
        correct_count += (label == pred).sum()
        
    print("accuracy (validation):", correct_count / sample_count)
