import torch
import os
import numpy as np
import glob
from data import MedSegDataset
import torch.nn as nn
import yaml
import pandas as pd
import torch.nn.functional as F
from unet_model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weighted_bce(outputs, targets, pos_weight):
    loss = F.binary_cross_entropy_with_logits(outputs, targets, pos_weight=pos_weight)
    return loss

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        # Flatten the tensors
        prediction = prediction.view(-1)
        target = target.view(-1)

        intersection = (prediction * target).sum()

        dice_coeff = (2. * intersection + self.epsilon) / (prediction.sum() + target.sum() + self.epsilon)
        return 1. - dice_coeff

if __name__ == "__main__":

    with open('config_seg.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    os.mkdir('./results/'+config['ProjectName'])
    # dataset and dataloader
    mri_files = config['mri_files']
    mri_files = np.array(glob.glob(mri_files))
    #shuffle mri_files
    np.random.seed(42)
    np.random.shuffle(mri_files)
    #split mri_files into train, validation and test in 70:15:15 ratio
    train_split = int(0.85 * len(mri_files))
    mri_files_train = mri_files[:train_split]
    mri_files_test = mri_files[train_split:]
    
    # Load the dataset
    train_dataset = MedSegDataset(config, mri_files_train, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = MedSegDataset(config, mri_files_test, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    data = next(iter(train_loader))
    print(len(train_loader), data[0].shape, data[1].shape)

    #model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #in_channels=3, out_channels=1, init_features=32, pretrained=False)

    #model.encoder1.enc1conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    model = UNet()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = True

    # Define optimizer
    pos_weight = torch.tensor([10]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dice_loss = DiceLoss()
    df_val = pd.DataFrame(columns=['epoch', 'dice'])
    df_train = pd.DataFrame(columns=['epoch', 'loss'])
    best_dice = 0

    for e in range(2000):
        # Train the model
        model.train()
        train_ls = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            input, target = data
            input = input.to(device)
            target = target.to(device)
            prediction = model(input)
            #prediction = torch.sigmoid(prediction)
            loss = criterion(prediction, target) + dice_loss(torch.sigmoid(prediction), target)
            #weighted_bce(torch.sigmoid(prediction), target, torch.tensor([10]).to(device)) #dice_loss(prediction, target)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            train_ls.append(loss.cpu().item())
        df_train = df_train.append({'epoch': e, 'loss': np.mean(np.array(train_ls))}, ignore_index=True)
        df_train.to_csv('./results/'+config['ProjectName']+'/train.csv', index=False)
        #save input, target and prediction
        np.save('./results/'+config['ProjectName']+f'/input_{e}.npy', input.cpu().detach().numpy())
        np.save('./results/'+config['ProjectName']+f'/target_{e}.npy', target.cpu().detach().numpy())
        model.eval()
        dices = []
        bce_loss = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                input, target = data
                input = input.to(device)
                target = target.to(device)
                prediction = model(input)
                #prediction = torch.sigmoid(prediction)
                loss = dice_loss(torch.sigmoid(prediction), target)
                loss2 = criterion(prediction, target)#weighted_bce(prediction, target, torch.tensor([10]).to(device))
                dice = 1-loss.cpu().item()
                dices.append(dice)
                bce_loss.append(loss2.cpu().item())
            df_val = df_val.append({'epoch': e, 'dice': np.mean(np.array(dices)), 'bce': np.mean(np.array(bce_loss))}, ignore_index=True)
            df_val.to_csv('./results/'+config['ProjectName']+ '/val.csv', index=False)
        
        if np.mean(np.array(dices)) > best_dice:
            best_dice = np.mean(np.array(dices))
            torch.save(model.state_dict(), './results/'+config['ProjectName']+ '/best_dice.pth')
        



