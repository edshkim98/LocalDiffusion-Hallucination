import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from data import MedDataset, MedDataset_png, MNIST, MvtecDatasetSR
import yaml
from medpy.io import load
from medpy.io import header
import glob
import timm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import idx2numpy

import timm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten the tensor for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":

    mode = 'mvtec'
    with open('config.yaml') as file:
        config_mri = yaml.load(file, Loader=yaml.FullLoader)

    config_data = {
    'mnist_path': './MNIST/raw/train-images-idx3-ubyte',
    'mnist_labels_path': './MNIST/raw/train-labels-idx1-ubyte',
    'mnist_test_path': './MNIST/raw/t10k-images-idx3-ubyte',
    'mnist_labels_test_path': './MNIST/raw/t10k-labels-idx1-ubyte'
    }

    #load mnist
    images = idx2numpy.convert_from_file(config_data['mnist_path'])
    labels = idx2numpy.convert_from_file(config_data['mnist_labels_path'])
    images_test = idx2numpy.convert_from_file(config_data['mnist_test_path'])
    labels_test = idx2numpy.convert_from_file(config_data['mnist_labels_test_path'])

    train_dataset = MNIST(config_data, images, labels, num=[0,1,2,3,4,5,6,7,8,9], train=False)#, max_file=1000)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = MNIST(config_data, images_test, labels_test, num=[0,1,2,3,4,5,6,7,8,9], train=False)#, max_file=1000)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = SimpleCNN()#timm.create_model('vgg16', pretrained=True, num_classes=10)
    #change the first layer to accept 1 channel
    # model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.to(device)


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 1000
    best_acc = 0
    df_loss = pd.DataFrame(columns=['epoch', 'train_loss', 'accuracy'])
    loss_lst = []
    for epoch in range(epochs):
        model.train()
        for i, (x, y, label) in enumerate(train_loader):
            x = x.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, label)
            loss.backward()
            loss_lst.append(loss.item())
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item()}')
        
        loss_mean = np.mean(loss_lst)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y, label in test_loader:
                x = x.to(device)
                label = label.to(device)
                output = model(x)
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == label).sum().item()
            print(f'Accuracy of the network on the test images: {100 * correct / total} %')
            if best_acc < 100 * correct / total:
                best_acc = 100 * correct / total
                torch.save(model.state_dict(), '/home/seunghki/mnist_az/vgg_mnist_cls_best_model.pth')

        df_loss = df_loss.append({'epoch': epoch, 'train_loss': loss_mean, 'accuracy': 100 * correct / total}, ignore_index=True)
        df_loss.to_csv('/home/seunghki/mnist_az/vgg_mnist_cls_loss.csv', index=False)

    print('Finished Training')