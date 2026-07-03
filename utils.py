"""Shared helpers used by both the training and testing entry points."""
import random

import numpy as np
import torch
import yaml

from ddpm import Unet


def load_config(path):
    with open(path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def build_model(config):
    """Build the conditional U-Net with the architecture matching config['data'].

    Train and test must construct the identical architecture, otherwise the
    checkpoints saved by the Trainer cannot be loaded back at test time.
    """
    data = config['data']
    dim = config['dim']
    if data == 'mri':
        return Unet(dim=dim, init_dim=dim, mode=data)
    if data == 'mnist':
        return Unet(dim=dim, init_dim=dim, dim_mults=(1, 2, 4), full_attn=(False, False, True), mode=data)
    if 'mvtec' in data:
        if data == 'mvtecSR':
            return Unet(dim=dim, init_dim=dim, dim_mults=(1, 2, 4), full_attn=(False, False, True),
                        channels=3, out_dim=3, mode=data)
        return Unet(dim=dim, init_dim=dim, channels=3, out_dim=3, mode=data)
    raise ValueError(f"unknown dataset '{data}', expected one of: mri, mnist, mvtec, mvtecSR")


def set_min_max_val(config):
    """Value range used to clamp the denoised prediction at every reverse step.

    For MRI the range is derived from the normalization statistics; for the
    natural-image datasets the images live in [0, 2] after loading (see data.py).
    Returns (min, max) or, for MRI, (min, max, min_t1).
    """
    data = config['data']
    if data == 'mri':
        if not config['translate_zero']:
            max_val = (4096 - config['mean_flair']) / config['std_flair']
            min_val = (0 - config['mean_flair']) / config['std_flair']
            min_val_t1 = (0 - config['mean_t1']) / config['std_t1']
        else:
            min_val2 = (0 - config['mean_flair']) / config['std_flair']
            min_val = 0.
            max_val = (4096 - config['mean_flair']) / config['std_flair']
            max_val = max_val + torch.abs(torch.tensor(min_val2))
            min_val_t1 = 0.
        return (min_val, max_val, min_val_t1)
    if (data == 'mnist') or ('mvtec' in data):
        return (0.0, 2.0)
    raise ValueError(f"unknown dataset '{data}', expected one of: mri, mnist, mvtec, mvtecSR")
