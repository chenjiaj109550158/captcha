# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
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
TEST_PATH = "/kaggle/input/captcha-hacker/test"

# try device = "cuda"
# and change your settings/accelerator to GPU if you want it to run faster


class Task1Dataset(Dataset):
    def __init__(self, data, root, transforms, return_filename=False):
        self.data = [
            sample for sample in data if sample[0].startswith("task1")]
        self.return_filename = return_filename
        self.root = root
        self.transforms = transforms

    def __getitem__(self, index):
        filename, label = self.data[index]
        img = cv2.imread(f"{self.root}/{filename}")
        denoised_img = dn.denoised_task1(img)
        denoised_mask = np.array([denoised_img.astype('float64')])/255
        img_d = np.zeros_like(img)
        for i in range(3):
            img_d[:, :, i] = img[:, :, i]*denoised_mask
        img = Image.fromarray(cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB))
        img = self.transforms(img)
        img = np.array(img)
        one_hot_vec = np.zeros(10)
        one_hot_vec[int(label)] = 1
        label = one_hot_vec

        if self.return_filename:
            return torch.FloatTensor(img), filename
        else:
            return torch.FloatTensor(img), label

    def __len__(self):
        return len(self.data)


class net_task1(nn.Module):
    def __init__(self):
        super(net_task1, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.d = nn.Dropout(p=0.2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.model_wo_fc(x)
        x = self.d(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':

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
            if random.random() < 1.1:
                train_data.append(row)
            else:
                val_data.append(row)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = transforms.Compose([transforms.Resize(
        224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    train_ds = Task1Dataset(train_data, root=TRAIN_PATH,
                            transforms=data_transforms)

    train_dl = DataLoader(train_ds, batch_size=200,
                          num_workers=4, drop_last=True, shuffle=True)

    val_ds = Task1Dataset(val_data, root=TRAIN_PATH,
                          transforms=data_transforms)
    val_dl = DataLoader(val_ds, batch_size=100, num_workers=4,
                        drop_last=False, shuffle=False)

    model = net_task1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    loss_hist = np.array([])
    is_final_train = 1
    for epoch in tqdm(range(100)):
        # print(f"Epoch [{epoch}]")
        model.train()
        for image, label in train_dl:
            image = image.to(device)
            label = label.to(device)

            pred = model(image)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist = np.append(loss_hist, loss.to('cpu').detach().numpy())
        sample_count = 0
        correct_count = 0
        if not is_final_train:
            model.eval()
            for image, label in val_dl:
                image = image.to(device)
                label = label.to(device)

                pred = model(image)
                loss = loss_fn(pred, label)

                pred = torch.argmax(pred, dim=1)
                label = torch.argmax(label, dim=1)

                sample_count += len(image)
                correct_count += (label == pred).sum()

            print("accuracy (validation):", correct_count / sample_count)

    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label='loss')
    plt.show()
    torch.save(model, 'model_t1.pt')
