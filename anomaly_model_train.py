"""Pre-compute the PatchCore memory bank used by the OOD detector at test time.

Run this once per dataset on *in-distribution* conditional images; test.py then
loads the resulting memory_bank_*.npy via config['memory_bank_path'].

Usage:
    python anomaly_model_train.py --mode mnist
    python anomaly_model_train.py --mode mvtec --mvtec_files './mvtec/leather/*/good/*.png' --obj leather
    python anomaly_model_train.py --mode mri --mri_files '/path/to/BRATS_png/normal/*flair.png'
"""
import argparse
import glob

import idx2numpy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader

from data import MedDataset_png, MNIST, MvtecDatasetSR
from models import PatchcoreModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-compute a PatchCore memory bank.')
    parser.add_argument('--mode', default='mnist', choices=['mnist', 'mvtec', 'mri'])
    parser.add_argument('--config', default='configs/mnist_test.yaml', help='config with dataset paths/statistics')
    parser.add_argument('--obj', default='leather', help='(mvtec) object name used in the output filename')
    parser.add_argument('--mvtec_files', default='./mvtec/leather/*/good/*.png')
    parser.add_argument('--mri_files', default='/path/to/BRATS_png/normal/*flair.png')
    parser.add_argument('--sampling_ratio', type=float, default=0.1, help='coreset subsampling ratio')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mode = args.mode

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # dataset of in-distribution conditional images
    if mode == 'mnist':
        images = idx2numpy.convert_from_file(config['mnist_path'])
        labels = idx2numpy.convert_from_file(config['mnist_labels_path'])
        train_dataset = MNIST(config, images, labels, num=3, train=False, max_file=300)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    elif mode == 'mvtec':
        print("MVTEC data")
        train_files = glob.glob(args.mvtec_files)
        np.random.seed(42)
        np.random.shuffle(train_files)
        print(len(train_files))
        train_dataset = MvtecDatasetSR(train_files, train=True, denoise=False, max_num=1000)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    else:
        mri_files = glob.glob(args.mri_files)
        np.random.seed(42)
        np.random.shuffle(mri_files)
        print(len(mri_files))
        train_dataset = MedDataset_png(config, mri_files, train=True, tumor=False)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print(len(train_loader))
    data = next(iter(train_loader))
    print(data[0].shape, data[1].shape)
    print(data[1].min(), data[1].max())
    print(data[0].min(), data[0].max())

    # PatchCore feature extractor
    backbone = "wide_resnet50_2"
    layers = ['layer2', 'layer3'] if 'resnet' in backbone else [1, 2]
    input_size = [84, 84] if mode == 'mnist' else [224, 224]
    patchcore = PatchcoreModel(input_size=input_size, layers=layers, backbone=backbone,
                               pre_trained=True, num_neighbors=9)
    patchcore.feature_extractor = patchcore.feature_extractor.to(device)
    patchcore.training = True

    # extract embeddings of the in-distribution images
    embeddings = []
    for i, data in enumerate(train_loader):
        _, input, *_ = data
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
        if mode != 'mri':
            if input.max() > 1.0:
                input = input / 2.0
        #normalize input using imagenet stats
        input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
        input = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input)
        input = input.to(device)

        embedding = patchcore(input)
        embeddings.append(embedding)

    print("All features extracted.")
    embeddings = torch.vstack(embeddings)

    print("Applying core-set subsampling to get the embedding.")
    patchcore.subsample_embedding(embeddings, args.sampling_ratio)
    print("Done.")

    if mode == 'mnist':
        out_name = 'memory_bank_mnist_train.npy'
    elif mode == 'mvtec':
        out_name = 'memory_bank_mvtec_{}.npy'.format('all' if args.obj == '*' else args.obj)
    else:
        out_name = 'memory_bank_mri_train.npy'
    np.save(out_name, patchcore.memory_bank.cpu().numpy())
    print("Saved memory bank to: ", out_name)
