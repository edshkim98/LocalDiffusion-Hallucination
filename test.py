import os
import glob
import numpy as np
import torch
import yaml
from ddpm import *
from data import *
import nibabel as nib
import idx2numpy
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    set_seed(42)
    
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # dataset and dataloader
    # dataset and dataloader
    file_path = './MNIST/raw/t10k-images-idx3-ubyte'
    label_path = './MNIST/raw/t10k-labels-idx1-ubyte'
    
    np.random.seed(42)
    mri_files = idx2numpy.convert_from_file(file_path)
    mri_labels = idx2numpy.convert_from_file(label_path)

    if config['ood']:
        ds_test = MNIST(config, mri_files, mri_labels, train=False, num=3, max_file=100) 
    else:
        ds_test = MNIST(config, mri_files, mri_labels, train=False, max_file=100) 

    dl_test = DataLoader(ds_test, batch_size = 1, shuffle = False, pin_memory = True, num_workers = cpu_count())

    data = next(iter(dl_test))
    print(len(ds_test), data[0].shape, data[1].shape)

    model = Unet(dim=config['dim'], init_dim=config['dim'])
    diffusion  = GaussianDiffusion(config, model, image_size=config['img_size'], timesteps=config['timestep'], objective = config['pred_objective'], auto_normalize=False)
    trainer = Trainer(config, diffusion, folder = None, train_batch_size=1)
    trainer.load('best')

    trainer.ema.ema_model.eval()

    with torch.inference_mode():
        lst = []
        lst_hr = []
        lst_pred = []
        lst_lr = []
        for i, data in enumerate(dl_test):
            hr, lr, cls = data
            hr = hr.to(device)
            lr = lr.to(device)
            cls = cls.to(device)
            
            if config['mask_cond']:
                if config['cond'] == 'IN':
                    mask = torch.zeros_like(lr)
                    length = mask.shape[-1]
                    mask[:,:,:,length//2:] = 1.0
                    lr = lr * mask
                else:
                    mask = torch.zeros_like(lr)
                    length = mask.shape[-1]
                    mask[:,:,:,length//2:] = 1.0
                    mask = 1.0 - mask
                    lr = lr * mask
            #Inference to get indistribution region
            out = trainer.ema.ema_model.sample(lr, batch_size=lr.shape[0], return_all_timesteps = False)
            #Inference to get OOD region
            # out = trainer.ema.ema_model.sample(lr, mask = (out, seg), batch_size=lr.shape[0], return_all_timesteps = False, background_mask = True)

            lst.append(torch.nn.MSELoss()(out, hr).cpu().detach().numpy())
            lst_hr.append(hr.cpu().detach().numpy())
            lst_pred.append(out.cpu().detach().numpy())
            lst_lr.append(lr.cpu().detach().numpy())

    #stack all the numpy arrays
    lst = np.array(lst)
    lst_hr = np.concatenate(np.array(lst_hr))
    lst_pred = np.concatenate(np.array(lst_pred))
    lst_lr = np.concatenate(np.array(lst_lr))

    np.save(f'hr_all.npy', lst_hr)
    np.save(f'lr_all.npy', lst_lr)
    np.save(f'pred_all.npy', lst_pred)

    ls = np.mean(np.array(lst))
    print("Test los: {:.4f}".format(ls))
