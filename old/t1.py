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

import my_models
import my_datasets

TRAIN_PATH = "E:/ml/hw5/train"
# TEST_PATH = "/kaggle/input/captcha-hacker/test"

# try device = "cuda" 
# and change your settings/accelerator to GPU if you want it to run faster







if __name__ == '__main__':

    
    net_task1 = my_models.get_model(1)

    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames[:3]:
            print(os.path.join(dirname, filename))
        if len(filenames) > 3:
            print("...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(0))
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

    data_transforms = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    
    Task1Dataset = my_datasets.get_dataset(1, train_data, root=TRAIN_PATH, transforms=data_transforms)
    train_ds = Task1Dataset

    # print(train_ds[0])
    


    train_dl = DataLoader(train_ds, batch_size=100, num_workers=4, drop_last=True, shuffle=True)

    val_ds = my_datasets.get_dataset(1, val_data, root=TRAIN_PATH, transforms=data_transforms)
    val_dl = DataLoader(val_ds, batch_size=100, num_workers=4, drop_last=False, shuffle=False)
    

    model = net_task1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    loss_hist = np.array([])
    
    for epoch in tqdm(range(100)):
        # print(f"Epoch [{epoch}]")
        model.train()
        for image, label in train_dl:
            image = image.to(device)
            label = label.to(device)
            print(label)
            exit()
            pred = model(image)
            loss = loss_fn(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist = np.append(loss_hist, loss.to('cpu').detach().numpy())
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


        
    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label='loss')
    plt.show()
    torch.save(model, 'model_t1.pt')