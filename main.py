import os
import glob
import numpy as np
import torch
import yaml
from ddpm import *
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    
    set_seed(42)

    with open('config_train.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model = Unet(dim=config['dim'], init_dim=config['dim'])
    diffusion  = GaussianDiffusion(config, model, image_size=config['img_size'], timesteps=config['timestep'], objective = config['pred_objective'], auto_normalize=False)
    trainer = Trainer(config, diffusion, folder=None, train_batch_size=64, save_and_sample_every = config['timestep']//4)

    trainer.train()
