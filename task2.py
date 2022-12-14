

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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


TRAIN_PATH = "E:/ml/hw5/train"
TEST_PATH = "/kaggle/input/captcha-hacker/test"
labels_map ='abcdefghijklmnopqrstuvwxyz0123456789'

# try device = "cuda" 
# and change your settings/accelerator to GPU if you want it to run faster


class Task2Dataset(Dataset):
    def __init__(self, data, root, transforms, return_filename=False):
        self.data = [sample for sample in data if sample[0].startswith("task2")]
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
        target = []
        for char in str(label):
                vec = [0] * 36 # num_class
                vec[labels_map.find(char)] = 1
                target += vec

        label = np.array(target)

        if self.return_filename:
            return torch.FloatTensor(img), filename
        else:
            return torch.FloatTensor(img), label

    def __len__(self):
        return len(self.data)

class net_task2(nn.Module):
    def __init__(self):
        super(net_task2, self).__init__()
        
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
        self.fc = nn.Linear(512, 72)

    def forward(self, x):
        
        # mean, std = x.mean([1,2]), x.std([1,2])
        # print(mean.shape)
        # exit()
        x = self.model_wo_fc(x)
        x = self.d(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = torch.sigmoid(self.fc(x))

        return x


if __name__ == '__main__':

    
# ***************
    

    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames[:3]:
            print(os.path.join(dirname, filename))
        if len(filenames) > 3:
            print("...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(0))
    # exit()
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
    train_ds = Task2Dataset(train_data, root=TRAIN_PATH, transforms=data_transforms)

    # print(train_ds[0][1].shape)
    # exit()
    


    train_dl = DataLoader(train_ds, batch_size=100, num_workers=4, drop_last=True, shuffle=True)

    val_ds = Task2Dataset(val_data, root=TRAIN_PATH, transforms=data_transforms)
    val_dl = DataLoader(val_ds, batch_size=100, num_workers=4, drop_last=False, shuffle=False)
    

    model = net_task2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MultiLabelSoftMarginLoss()

    loss_hist = np.array([])
    
    for epoch in tqdm(range(100)):
        # print(f"Epoch [{epoch}]")
        model.train()
        for image, label in train_dl:
            image = image.to(device)
            label = label.to(device)

            pred = model(image)
            # print(label.dtype)
            # print(pred.dtype)
            # exit()
            
            loss = loss_fn(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist = np.append(loss_hist, loss.to('cpu').detach().numpy())
            # print(loss.to('cpu').detach().numpy())
        sample_count = 0
        correct_count = 0
        model.eval()
        for image, label in val_dl:
            image = image.to(device)
            label = label.to(device)
            
            pred = model(image).to('cpu').detach().numpy()
            # for p in pred:
            #     print(p)
            label = label.to('cpu').detach().numpy()
            # print(pred.shape)
            pred_a = []
            for i in pred:
                # print(output)
                # exit()
                y_p= ''
                # print(output.to('cpu').detach().numpy())
                # exit()
                idx = np.argmax(i[:36])
                y_p += str(labels_map[idx])
                idx = np.argmax(i[36:])
                y_p += str(labels_map[idx])
                pred_a.append(y_p)
            label_a = []
            for i in label:
                # print(output)
                # exit()
                y_p= ''
                # print(output.to('cpu').detach().numpy())
                # exit()
                idx = np.argmax(i[:36])
                y_p += str(labels_map[idx])
                idx = np.argmax(i[36:])
                y_p += str(labels_map[idx])
                label_a.append(y_p)
            
            # print(torch.argmax(pred[0:36], dim=1))
            pred = np.array(pred_a)
            label = np.array(label_a)

            sample_count += len(image)
            correct_count += (label == pred).sum()
            
        print("accuracy (validation):", correct_count / sample_count)
        

        
    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label='loss')
    plt.show()
    torch.save(model, 'model_t2.pt')
