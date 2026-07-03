"""Train a small CNN digit classifier on MNIST (used for the misclassification evaluation)."""
import os

import idx2numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import MNIST
from models import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

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

    train_dataset = MNIST(config_data, images, labels, num=[0,1,2,3,4,5,6,7,8,9], train=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = MNIST(config_data, images_test, labels_test, num=[0,1,2,3,4,5,6,7,8,9], train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    os.makedirs('./results', exist_ok=True)

    model = SimpleCNN()
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
                torch.save(model.state_dict(), './results/mnist_cls_best_model.pth')

        df_loss = pd.concat([df_loss, pd.DataFrame([{'epoch': epoch, 'train_loss': loss_mean,
                                                     'accuracy': 100 * correct / total}])], ignore_index=True)
        df_loss.to_csv('./results/mnist_cls_loss.csv', index=False)

    print('Finished Training')
