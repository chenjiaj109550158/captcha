
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
import denoise as dn

TRAIN_PATH = "E:/ml/hw5/train"
TEST_PATH = "E:/ml/hw5/test"

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
        img = cv2.imread(f"{self.root}/{filename}")
        # print(img.dtype)
        denoised_img = dn.denoised_task2(img)
        denoised_mask = np.array([denoised_img.astype('float64')])/255
        
        img_d = np.zeros_like(img)
        for i in range(3):
            img_d[:,:,i] = img[:,:,i]*denoised_mask
        img = Image.fromarray(cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB))
        # img = Image.open(f"{self.root}/{filename}")
        img = self.transforms(img)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net_task2().to(device)
    model= torch.load('model_t2_den.pt')

    test_data = []
    with open(f'{TEST_PATH}/sample_submission.csv', newline='') as csvfile:
        for row in csv.reader(csvfile, delimiter=','):
            test_data.append(row)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    data_transforms = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test_ds = Task2Dataset(test_data, root=TEST_PATH, return_filename=True, transforms=data_transforms)
    test_dl = DataLoader(test_ds, batch_size=100, num_workers=4, drop_last=False, shuffle=False)


    if os.path.exists('submission.csv'):
        csv_writer = csv.writer(open('submission.csv', 'a', newline=''))
    else:
        csv_writer = csv.writer(open('submission.csv', 'w', newline=''))
        csv_writer.writerow(["filename", "label"])


    model.eval()
    for image, filenames in test_dl:
        # image = image.to(device)
        
        # pred = model(image)
        # pred = torch.argmax(pred, dim=1)
        image = image.to(device)
        
        pred = model(image).to('cpu').detach().numpy()

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
        
        # print(torch.argmax(pred[0:36], dim=1))
        pred = np.array(pred_a)
        

        for i in range(len(filenames)):
            csv_writer.writerow([filenames[i], str(pred[i].item())])

    # for filename, _ in test_data:
    #     if filename.startswith("task2") or filename.startswith("task3"):
    #         csv_writer.writerow([filename, -1])