"""Evaluate a trained diffusion model with the local-diffusion (branch + fusion) sampling.

The script
  1. loads a trained checkpoint (saved by train.py),
  2. detects the OOD region of each conditional image with PatchCore
     (or an optional pre-trained segmentation model for MRI),
  3. runs the branched reverse diffusion and fuses the IND/OOD predictions,
  4. saves the inputs, predictions and OOD masks as .npy files.

Example:
    python test.py --config configs/mnist_test.yaml
"""
import argparse
import glob
import os
import time
from multiprocessing import cpu_count
from pathlib import Path

import idx2numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from data import MNIST, MedDataset_png, MvtecDatasetSR
from ddpm import GaussianDiffusion, Trainer
from utils import build_model, load_config, set_min_max_val, set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Test the diffusion model with local diffusion sampling.')
    parser.add_argument('--config', default='configs/mnist_test.yaml', help='path to the test config yaml')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def build_test_dataset(config):
    """Return the test dataset: OOD samples if config['ood'] else IND samples."""
    if config['data'] == 'mnist':
        file_path = config.get('mnist_test_path', './MNIST/raw/t10k-images-idx3-ubyte')
        label_path = config.get('mnist_labels_test_path', './MNIST/raw/t10k-labels-idx1-ubyte')
        images = idx2numpy.convert_from_file(file_path)
        labels = idx2numpy.convert_from_file(label_path)
        if config['ood']:
            #OOD digits, e.g. testing a model trained on 8s with 3s
            return MNIST(config, images, labels, train=False, num=config['anomaly_name'], max_file=100)
        return MNIST(config, images, labels, train=False, num=[8], max_file=100)

    if config['data'] == 'mri':
        print("Data: Brain")
        np.random.seed(42)
        mri_files = np.array(glob.glob(config['mri_files']))
        np.random.shuffle(mri_files)
        if config['ood']:
            train_split = int(0.5 * len(mri_files))
            mri_files_test = mri_files[train_split:]
            print(len(mri_files_test))
            return MedDataset_png(config, mri_files_test, train=False, tumor=True, mode='t1')
        train_split = int(0.7 * len(mri_files))
        mri_files_test = mri_files[train_split:]
        return MedDataset_png(config, mri_files_test, train=False, tumor=False)

    if 'mvtec' in config['data']:
        print("Data: MVTec {}".format(mvtec_object(config)))
        np.random.seed(42)
        mri_files = np.array(glob.glob(config['mvtec_path']))
        np.random.shuffle(mri_files)
        if config['ood']:
            print(config['anomaly_name'], len(mri_files))
            return MvtecDatasetSR(mri_files, train=False, mode=config['anomaly_name'])
        return MvtecDatasetSR(mri_files, train=True)

    raise ValueError(f"unknown dataset '{config['data']}', expected one of: mri, mnist, mvtec, mvtecSR")


def mvtec_object(config):
    """Name of the MVTec object category, e.g. 'transistor'."""
    if config.get('mvtec_object'):
        return config['mvtec_object']
    parts = Path(config['mvtec_path']).parts
    for i, part in enumerate(parts):
        if part == 'mvtec' and i + 1 < len(parts):
            return parts[i + 1]
    raise ValueError("could not infer the MVTec object from config['mvtec_path']; set config['mvtec_object']")


def load_patchcore_detector(config):
    """PatchCore anomaly detector used to segment the OOD region of the conditional image.

    Its memory bank must be pre-computed on the (in-distribution) conditional-image
    domain with anomaly_model_train.py and pointed to by config['memory_bank_path'].
    """
    from models import PatchcoreModel  # imported lazily: requires anomalib

    backbone = 'wide_resnet50_2'
    layers = ['layer2', 'layer3'] if 'resnet' in backbone else [2, 3]
    img_size = 84 if config['data'] == 'mnist' else 224

    patchcore = PatchcoreModel(input_size=[img_size, img_size], layers=layers,
                               backbone=backbone, pre_trained=True, num_neighbors=9)
    bank_path = config.get('memory_bank_path')
    if bank_path is None:
        raise ValueError("config['memory_bank_path'] must point to the PatchCore memory bank (.npy) "
                         "pre-computed with anomaly_model_train.py")
    patchcore.memory_bank = torch.from_numpy(np.load(bank_path)).to(device)
    patchcore.training = False
    patchcore.feature_extractor = patchcore.feature_extractor.to(device)
    patchcore.feature_extractor.eval()
    return patchcore


def load_seg_detector(config):
    """Optional MRI-only alternative to PatchCore: a pre-trained tumor segmentation U-Net."""
    from unet_model import UNet

    seg_model = UNet()
    seg_model.load_state_dict(torch.load(config['ood_detector']['seg_model']))
    seg_model = seg_model.to(device)
    seg_model.eval()
    return seg_model


def soft_ood_mask(anomaly_map, threshold, clip_min):
    """Turn an anomaly map into a soft OOD-probability mask plus its binarization.

    Values above `threshold` are OOD (mask 1); values below fade towards 0 so that
    'some information' about the borderline region is kept for the IND branch.
    """
    binary_mask = (anomaly_map > threshold).float()
    map_pred = torch.clip(anomaly_map, min=clip_min, max=threshold)
    mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
    mask_pred = mask_pred ** 2
    return mask_pred, binary_mask


def all_ind_mask(anomaly_map):
    """No OOD detected: an all-ones mask makes sample() fall back to the vanilla reverse process."""
    return torch.ones_like(anomaly_map), torch.ones_like(anomaly_map)


def compute_ood_mask(config, anomaly_map):
    """Map a PatchCore anomaly map to (soft mask, binary mask).

    The entry conditions and thresholds below are the per-dataset values calibrated
    for the experiments in the paper; if the anomaly score is too low the whole
    image is treated as in-distribution.
    """
    amap = anomaly_map.cpu()
    peak = amap.max()

    if config['data'] == 'mnist':
        if config['mnist_cls'] == '8to3':
            if peak > 37.0:
                if peak > 44:
                    threshold = 41.7
                elif peak > 40.0:
                    threshold = 38.2
                else:
                    threshold = 35.0
                return soft_ood_mask(amap, threshold, threshold - amap.std())
            return all_ind_mask(amap)
        if config['mnist_cls'] == '8to5':
            if peak > 58.5:
                if peak > 71.0:
                    threshold = 61.0
                elif peak > 65:
                    threshold = 57.0
                else:
                    threshold = 55.0
                return soft_ood_mask(amap, threshold, threshold - amap.std())
            return all_ind_mask(amap)
        raise ValueError(f"unknown mnist_cls '{config['mnist_cls']}'")

    if config['data'] == 'mri':
        if 't12flair' in config['ProjectName']:
            if peak > 43:
                if peak > 60:
                    threshold = peak - 12
                elif peak > 51:
                    threshold = 47
                elif peak > 48.5:
                    threshold = 44
                else:
                    threshold = 42
                return soft_ood_mask(amap, threshold, threshold - amap.std())
            return all_ind_mask(amap)
        if 'flair2t1' in config['ProjectName']:
            if peak > 43:
                if peak > 60:
                    threshold = 47
                elif peak > 50:
                    threshold = 43
                else:
                    threshold = 42
                return soft_ood_mask(amap, threshold, threshold - amap.std())
            return all_ind_mask(amap)
        raise ValueError("for MRI, config['ProjectName'] must contain 't12flair' or 'flair2t1' "
                         "to select the calibrated OOD thresholds")

    if 'mvtec' in config['data']:
        obj = mvtec_object(config)
        if obj == 'transistor':
            if peak > 32:
                if peak > 40.0:
                    threshold = 33.5
                elif peak > 36.8:
                    threshold = peak - 2 * amap.std()
                elif peak > 35.0:
                    threshold = peak - 1 * amap.std()
                else:
                    threshold = 29.5
                return soft_ood_mask(amap, threshold, threshold - 0.5 * amap.std())
            return all_ind_mask(amap)
        if obj == 'toothbrush':
            if peak > 35:
                threshold = 40.0 if peak > 49 else 28.0
                return soft_ood_mask(amap, threshold, amap.min())
            return all_ind_mask(amap)
        if obj == 'grid':
            if peak > 27:
                if peak > 40:
                    threshold = 35.0
                elif peak > 35.0:
                    threshold = 30.0
                else:
                    threshold = 26.5
                return soft_ood_mask(amap, threshold, amap.min())
            return all_ind_mask(amap)
        raise ValueError(f"no calibrated OOD thresholds for MVTec object '{obj}' "
                         "(available: transistor, toothbrush, grid)")

    raise ValueError(f"unknown dataset '{config['data']}'")


def prepare_detector_input(config, lr):
    """Normalize/resize the conditional image the way the PatchCore backbone expects."""
    lr_ad = lr.repeat(1, 3, 1, 1) if lr.shape[1] != 3 else lr.clone()

    if config['data'] == 'mri':
        #denormalize back to raw intensities, then rescale to [0, 1]
        if config['translate_zero']:
            mini = (0 - config['mean_t1']) / config['std_t1']
            lr_ad = lr_ad - torch.abs(torch.tensor(mini))
        lr_ad = lr_ad[:, 0] * config['std_t1'] + config['mean_t1']
        lr_ad = lr_ad / 4096.0
        lr_ad = lr_ad.repeat(1, 3, 1, 1)

    if ('mvtec' in config['data']) or (config['data'] == 'mnist'):
        if lr_ad.shape[1] == 1:
            lr_ad = lr_ad.repeat(1, 3, 1, 1)
        if lr_ad.max() > 1.0:
            print("Normalize LR AD")
            lr_ad = lr_ad / 2
        img_size = 224 if 'mvtec' in config['data'] else 84
        lr_ad = F.interpolate(lr_ad, size=(img_size, img_size), mode='bilinear', align_corners=False)

    lr_ad = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(lr_ad)
    return lr_ad


if __name__ == "__main__":
    args = parse_args()
    print("Device: ", device)
    set_seed(args.seed)

    config = load_config(args.config)
    print(config['ProjectName'])

    out_dir = os.path.join(config['Results'], config['ProjectName'].strip('/'), 'test_outputs')
    os.makedirs(out_dir, exist_ok=True)

    # dataset and dataloader
    ds_test = build_test_dataset(config)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=cpu_count())

    data = next(iter(dl_test))
    print(len(ds_test), data[0].shape, data[1].shape)

    # model
    min_max_val = set_min_max_val(config)
    model = build_model(config)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", pytorch_total_params)

    if config['ddim_timestep'] == False:
        config['ddim_timestep'] = None
    diffusion = GaussianDiffusion(config, model, image_size=config['img_size'], timesteps=config['timestep'],
                                  beta_schedule=config['scheduler'], objective=config['pred_objective'],
                                  auto_normalize=False, sampling_timesteps=config['ddim_timestep'])
    trainer = Trainer(config, diffusion, folder=None, train_batch_size=1)
    trainer.load('best' + str(config['train_phase']))

    #optional PatchCore classifier used by the adaptive fusion step (config['classifier'])
    trainer.ema.ema_model.call_classifier()
    trainer.ema.ema_model.eval()

    #OOD detector used to partition the conditional image
    patchcore, seg_model = None, None
    if config['ood_AD']:
        if config['data'] == 'mri' and config['ood_detector']['seg']:
            seg_model = load_seg_detector(config)
        else:
            patchcore = load_patchcore_detector(config)

    #test loop
    with torch.inference_mode():
        lst, lst_hr, lst_pred, lst_lr, lst_masks, lst_defect_name, times = [], [], [], [], [], [], []
        print("Test start!")
        for i, data in enumerate(dl_test):
            if len(data) == 3:
                hr, lr, cls = data
            else:
                hr, lr, cls, defect = data
                lst_defect_name.append(defect)
            hr, lr, cls = hr.to(device), lr.to(device), cls.to(device)

            mask_pred, binary_mask = None, None
            if config['ood_AD']:
                print("Segmenting OOD from conditional image...")
                if seg_model is not None:
                    #segment using the pre-trained segmentation model instead of PatchCore
                    mini = (0 - config['mean_t1']) / config['std_t1']
                    lr_ad = lr - torch.abs(torch.tensor(mini))
                    print(lr_ad.min(), lr_ad.max())
                    seg_out = seg_model(lr_ad)
                    mask_pred = nn.Sigmoid()(seg_out).detach().cpu()
                    binary_mask = (mask_pred > 0.5).float()
                    mask_pred = binary_mask
                else:
                    lr_ad = prepare_detector_input(config, lr)
                    print(lr_ad.shape)
                    pred_anomalymap = patchcore(lr_ad.to(device))
                    anomaly_map, pred_score = pred_anomalymap["anomaly_map"], pred_anomalymap["pred_score"]
                    if ('mvtec' in config['data']) or (config['data'] == 'mnist'):
                        anomaly_map = F.interpolate(anomaly_map, size=(config['img_size'], config['img_size']),
                                                    mode='bilinear', align_corners=False)
                    print("Anomaly score: ", anomaly_map.max())
                    mask_pred, binary_mask = compute_ood_mask(config, anomaly_map)

                if config['data'] == 'mri':
                    cls[cls > 0.0] = 1.0

                #### Alternatively, manually partition the conditional image (motivational exp. 1) ####
                #### by uncommenting the lines below ####
                # mask_pred = torch.zeros_like(anomaly_map.cpu())
                # mask_pred[:, :, :, :7] = 1.0
                # binary_mask = mask_pred

                lst_masks.append(mask_pred.cpu().detach().numpy())

            print("LR Min {} Max {}, HR Min {} Max {}".format(lr.min(), lr.max(), hr.min(), hr.max()))
            if config['ood_AD']:
                #the soft mask drives the branching; the binary mask is used when branching is off
                mask = mask_pred if config['branch_out'] else binary_mask
                mask = mask.to(device)
            elif (config['data'] == 'mnist') and (config['branch_out'] == True):
                #fixed half-image partition used in the motivational MNIST experiment
                mask = torch.zeros_like(lr)
                mask[:, :, :, 14:] = 1.0
                mask = 1.0 - mask
                print("MNIST mask generated")
            else:
                mask = None

            start = time.time()
            out = trainer.ema.ema_model.sample(lr, hr, batch_size=lr.shape[0],
                                               return_all_timesteps=config['return_all_timesteps'],
                                               return_all_outputs=config['return_all_out'],
                                               mask=mask, min_max_val=min_max_val)
            times.append(time.time() - start)

            lst.append(torch.nn.MSELoss()(out[:, [-1]].cpu(), hr.cpu()))
            lst_hr.append(hr.cpu().detach().numpy())
            lst_pred.append(out.cpu().detach().numpy())
            lst_lr.append(lr.cpu().detach().numpy())

    #stack and save all outputs
    lst = np.array(lst)
    lst_hr = np.concatenate(np.array(lst_hr))
    lst_pred = np.concatenate(np.array(lst_pred))
    lst_lr = np.concatenate(np.array(lst_lr))
    np.save(os.path.join(out_dir, 'hr_all.npy'), lst_hr)
    np.save(os.path.join(out_dir, 'lr_all.npy'), lst_lr)
    np.save(os.path.join(out_dir, 'pred_all.npy'), lst_pred)
    if 'mvtec' in config['data']:
        lst_defect_name = np.concatenate(np.array(lst_defect_name))
        np.save(os.path.join(out_dir, 'defect_name.npy'), lst_defect_name)
    if config['ood_AD']:
        print(lst_masks[0].shape)
        lst_masks = np.concatenate(np.array(lst_masks))
        np.save(os.path.join(out_dir, 'ad_masks.npy'), lst_masks)

    print("Saved outputs to: ", out_dir)
    print("Test loss: {:.4f}".format(np.mean(np.array(lst))))
    print("Average sampling time: {:.4f}".format(np.mean(np.array(times))))
