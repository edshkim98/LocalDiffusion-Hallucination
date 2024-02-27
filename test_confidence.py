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

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import read_image
from anomalib.pre_processing.transforms import Denormalize
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks

from anomalib.models.components import DynamicBufferModule, FeatureExtractor, KCenterGreedy
from anomalib.models.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler
from patchcore import PatchcoreModel
from torchvision.transforms import GaussianBlur

import timm

class AverageFilter(nn.Module):
    def __init__(self, kernel_size, rgb = False):
        super(AverageFilter, self).__init__()
        # Create a kernel where each value is 1/(kernel_size*kernel_size)
        # This ensures that the sum of the kernel equals 1
        kernel = torch.ones((1, 1, kernel_size, kernel_size)) / (kernel_size * kernel_size)
        # Repeat the kernel for the number of input channels
        # Assuming grayscale images, for RGB, you need 3 input channels and repeat the kernel 3 times accordingly
        if rgb:
            kernel = kernel.repeat(1, 3, 1, 1)
        else:
            kernel = kernel.repeat(1, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=kernel_size//2, bias=False, padding_mode='reflect')
        self.conv.weight = self.weight

    def forward(self, x):
        x = self.conv(x)
        return x
    
def normalize_neg_one_to_one(x):
    return (x - 0.5) / 0.5

def unnormalize_zero_to_one(x):
    return x * 0.5 + 0.5

def gaussian_pdf(x, mean, std, eps=1e-4):
    variance = std ** 2
    return torch.exp(-((x - mean) ** 2) / (2 * variance + eps)) / torch.sqrt(2 * np.pi * variance + eps)
        
def normalize_image(img):
    # Convert img to float if necessary
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255

    # Define mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Ensure the image is in HxWxC format
    if img.ndim == 2:  # Grayscale
        img = np.expand_dims(img, axis=-1)
        mean = mean[0]  # Use only the first channel of mean and std
        std = std[0]

    # Normalize image
    normalized_img = (img - mean) / std

    return normalized_img

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    print("Device: ", device)
    set_seed(0)
    
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    print(config['ProjectName'])
    # dataset and dataloader
    # dataset and dataloader
    file_path = './MNIST/raw/t10k-images-idx3-ubyte'
    label_path = './MNIST/raw/t10k-labels-idx1-ubyte'
    
    np.random.seed(42)
    mri_files = idx2numpy.convert_from_file(file_path)
    mri_labels = idx2numpy.convert_from_file(label_path)
    if config['ood']:
        if config['data'] == 'mnist':
            ds_test = MNIST(config, mri_files, mri_labels, train=False, num=[3], max_file=1000) 
        elif config['data'] == 'mri':
            print("Data: Brain")
            np.random.seed(42)
            mri_files = np.array(glob.glob(config['mri_files']))
            np.random.shuffle(mri_files)

            #split mri_files into train, validation and test in 70:15:15 ratio
            train_split = int(0.8 * len(mri_files))
            mri_files_test = mri_files[train_split:]

            print(len(mri_files_test)) 
            ds_test = MedDataset_png(config, mri_files_test, train=False, tumor=True, mode='t1')
        elif 'mvtec' in config['data']:
            print("Data: MVTec {}".format(config['mvtec_path'].split('/')[-2]))
            np.random.seed(42)
            mri_files = config['mvtec_path']
            mri_files = np.array(glob.glob(mri_files))
            np.random.shuffle(mri_files)
            print(config['anomaly_name'], len(mri_files))
            
            if 'mvtecSR' == config['data']:
                ds_test = MvtecDatasetSR(mri_files, train=False, mode=config['anomaly_name'])
            elif config['data'] == 'mvtecGray':
                ds_test = MvtecDatasetGray(mri_files, train=False, mode=config['mvtec_path'].split('/')[-2], denoise=config['denoise'])
            else:
                ds_test = MvtecDatasetSR(mri_files, train=False, mode=config['anomaly_name'], denoise=config['denoise'])
    else:
        if config['data'] == 'mnist':
            ds_test = MNIST(config, mri_files, mri_labels, train=False, max_file=100) 
        elif config['data'] == 'mri':
            mri_files = np.array(glob.glob(config['mri_files']))
            train_split = int(0.7*len(mri_files))
            mri_files_test = mri_files[train_split:]
            ds_test = MedDataset_png(config, mri_files_test, train=False, tumor=False)
        elif 'mvtec' in config['data']:
            print("Data: MVTec")
            np.random.seed(42)
            mri_files = config['mvtec_path']
            mri_files = np.array(glob.glob(mri_files))
            np.random.shuffle(mri_files)
            if 'mvtecSR' == config['data']:
                ds_test = MvtecDatasetSR(mri_files, train=True)
            else:
                ds_test = MvtecDataset(mri_files, train=True)

    dl_test = DataLoader(ds_test, batch_size = 1, shuffle = False, pin_memory = True, num_workers = cpu_count())

    data = next(iter(dl_test))
    print(len(ds_test), data[0].shape, data[1].shape)

    if config['data'] == 'mri': 
        if not config['translate_zero']:
            max_val = (4096-config['mean_flair'])/config['std_flair']
            min_val = (0-config['mean_flair'])/config['std_flair']
            min_val_t1 = (0-config['mean_t1'])/config['std_t1']
        else:
            min_val2 = (0-config['mean_t1'])/config['std_t1'] 
            min_val = 0.
            max_val = (4096-config['mean_t1'])/config['std_t1']
            max_val = max_val + torch.abs(torch.tensor(min_val2))
            min_val_t1 = 0.
        min_max_val = (min_val, max_val, min_val_t1) 
    elif config['data'] == 'mnist':
        min_val = 0.0
        max_val = 2.0
        min_max_val = (min_val, max_val) 
    elif 'mvtec' in config['data']:
        min_val = 0.
        max_val = 2.0
        min_max_val = (min_val, max_val)

    if config['data'] == 'mri':
         model = Unet(dim=config['dim'], init_dim=config['dim'], mode=config['data'])
    elif 'mvtec' in config['data']:
        if config['data'] == 'mvtecSR':
            channels = 3
            out_dim = 3
        elif config['data'] == 'mvtecGray':
            channels = 1
            out_dim = 1
        else:
            channels = 3
            out_dim = 3
        if config['data'] == 'mvtecSR':
            model = Unet(dim=config['dim'], init_dim=config['dim'], dim_mults = (1, 2, 4), full_attn = (False, False, True), channels = channels, out_dim=out_dim, mode=config['data'])
        else:
            model = Unet(dim=config['dim'], init_dim=config['dim'], channels = channels, out_dim=out_dim, mode=config['data'])         
    elif config['data'] == 'mnist':
         model = Unet(dim=config['dim'], init_dim=config['dim'], dim_mults = (1, 2, 4), full_attn = (False, False, True), mode=config['data'])
    
    if config['ddim_timestep'] == False:
        config['ddim_timestep'] = None
    diffusion  = GaussianDiffusion(config, model, image_size=config['img_size'], timesteps=config['timestep'], beta_schedule=config['scheduler'], objective = config['pred_objective'], auto_normalize=False, sampling_timesteps=config['ddim_timestep'])
    trainer = Trainer(config, diffusion, folder = None, train_batch_size=1)
    train_phase = config['train_phase']
    trainer.load('best'+str(train_phase))

    print("Loading classifier")
    trainer.ema.ema_model.call_classifier()
    print("Classifier loaded")
    
    trainer.ema.ema_model.eval()

    if config['ood_AD']:
        MODEL = "patchcore"  # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
        CONFIG_PATH = './anomalib/' + f"src/anomalib/models/{MODEL}/config.yaml"
        # pass the config file to model, callbacks and datamodule
        config_ad = get_configurable_parameters(config_path=CONFIG_PATH)

        backbone = 'wide_resnet50_2'
        if 'resnet' in backbone:
            if 'mvtec' in config['data']:
                layers = ['layer2', 'layer3']
            else:
                layers = ['layer2', 'layer3']
        else:
            layers = [2, 3]
        if config['data'] == 'mnist':
            img_size = 84
        patchcore = PatchcoreModel(input_size = [224, 224], layers = layers,backbone= backbone, pre_trained= True, num_neighbors= 9)
        if config['data'] == 'mnist':
            pretrained = np.load('./memory_bank_mnist.npy')
        elif 'mvtec' in config['data']: 
            pretrained = np.load('./memory_bank_mvtec_{}.npy'.format(config['mvtec_path'].split('/')[2]))
        else:
            pretrained = np.load('./memory_bank_mri.npy')
        patchcore.memory_bank = torch.from_numpy(pretrained)#.to(device)
        patchcore.training = False
        patchcore.feature_extractor = patchcore.feature_extractor.cpu()
        patchcore.feature_extractor.eval()


    with torch.inference_mode():
        lst = []
        lst_hr = []
        lst_pred = []
        lst_lr = []
        lst_x_starts = []
        lst_confidence = []
        lst_masks = []
        lst_features = []
        lst_stats_mean = []
        lst_stats_std = []
        lst_lr_mask_ood = []
        lst_defect_name = []
        print("Test start!")
        for i, data in enumerate(dl_test):
            if len(data) == 3:
                hr, lr, cls = data
                hr = hr.to(device)
                lr = lr.to(device)
                cls = cls.to(device)
            elif len(data) == 4:
                hr, lr, cls, defect = data
                hr = hr.to(device)
                lr = lr.to(device)
                cls = cls.to(device)
                lst_defect_name.append(defect)


            if config['ood_AD']:
                print("OOD AD")
                if lr.shape[1] != 3:
                    lr_ad = lr.repeat(1, 3, 1, 1)
                else:
                    lr_ad = lr.clone()

                #normalize image using imagenet mean and std
                if config['data'] == 'mri':
                    #denormalize first
                    if config['translate_zero']:
                        mini = (0-config['mean_flair'])/config['std_flair']
                        lr_ad = lr_ad - torch.abs(torch.tensor(mini))
                    lr_ad = lr_ad[:,0]*config['std_flair'] + config['mean_flair']
                    lr_ad = lr_ad/4096.0
                    lr_ad = lr_ad.repeat(1, 3, 1, 1)
                if ('mvtec' in config['data']) or (config['data'] == 'mnist'):
                    if lr_ad.shape[1] == 1:
                        lr_ad = lr_ad.repeat(1, 3, 1, 1)
                    if lr_ad.max() > 1.0:
                        print("Normalize LR AD")
                        lr_ad = lr_ad / 2
                    if 'mvtec' in config['data']:
                        img_size = 224
                    else:
                        img_size = 84
                    lr_ad = F.interpolate(lr_ad, size=(img_size, img_size), mode='bilinear', align_corners=False)
                print(lr_ad.shape)
                lr_ad = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(lr_ad)

                pred_anomalymap = patchcore(lr_ad.cpu())
                anomaly_map, pred_score = pred_anomalymap["anomaly_map"], pred_anomalymap["pred_score"]
                if ('mvtec' in config['data']) or (config['data'] == 'mnist'):
                    anomaly_map = F.interpolate(anomaly_map, size=(config['img_size'], config['img_size']), mode='bilinear', align_corners=False)
                    # anomaly_map = AverageFilter(kernel_size=3)(anomaly_map)
                print("Anomaly score: ", anomaly_map.max())
                if config['data'] == 'mnist':
                    if anomaly_map.max() > 37.0:
                        if anomaly_map.max() > 41.0:#44
                            threshold = anomaly_map.max()-0.72*anomaly_map.std() #41.7
                        elif anomaly_map.max() > 40.0:
                            threshold = anomaly_map.max()-0.3*anomaly_map.std() #38.2
                        else:
                            threshold = 35.0
                        binary_mask = (anomaly_map.cpu() > threshold).float()
                        map_pred = torch.clip(anomaly_map.cpu(), min=threshold-anomaly_map.std(), max=threshold)
                        mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                        mask_pred = mask_pred **2
                    else:
                        mask_pred = torch.ones_like(anomaly_map.cpu())
                        binary_mask = torch.ones_like(anomaly_map.cpu())
                if config['data'] == 'mri':
                        if 't12flair' in config['ProjectName']:
                            if anomaly_map.max() > 43:
                                if anomaly_map.max() > 50:
                                    threshold = anomaly_map.max()-7
                                else:
                                    threshold = 43
                                binary_mask = (anomaly_map.cpu() > threshold).float()
                                map_pred = torch.clip(anomaly_map.cpu(), min=threshold-anomaly_map.std(), max=threshold)
                                mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                                mask_pred = mask_pred **2
                            else:
                                mask_pred = torch.ones_like(anomaly_map.cpu())
                                binary_mask = torch.ones_like(anomaly_map.cpu())

                        elif 'flair2t1' in config['ProjectName']:
                            if anomaly_map.max() > 47:
                                if anomaly_map.max() > 50:
                                    threshold = anomaly_map.max()-3
                                else:
                                    threshold = 45.5
                                binary_mask = (anomaly_map.cpu() > threshold).float()
                                map_pred = torch.clip(anomaly_map.cpu(), min=threshold-anomaly_map.std(), max=threshold)
                                mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                                mask_pred = mask_pred **2
                            else:
                                mask_pred = torch.ones_like(anomaly_map.cpu())
                                binary_mask = torch.ones_like(anomaly_map.cpu())

                        cls[cls>0.0] = 1.0
                        #mask_pred = cls.cpu()
                        #binary_mask = cls.cpu()
                
                if 'mvtec' in config['data']:
                    if config['mvtec_path'].split('/')[2] == 'transistor': #64
                        print("TRANSISTOR!!")
                        if anomaly_map.max() > 27:
                            if anomaly_map.max() > 37.0:
                                threshold = 31.0#32 #anomaly_map.max()-5.0
                            elif anomaly_map.max() > 33.0:

                                threshold = 29.5 #anomaly_map.max()-4.0
                            else:
                                threshold = 26.8 #anomaly_map.max()-2.5
                            binary_mask = (anomaly_map.cpu() > threshold).float()
                            map_pred = torch.clip(anomaly_map.cpu(), min=anomaly_map.min(), max=threshold)
                            mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                            mask_pred = mask_pred**2
                        else:
                            mask_pred = torch.ones_like(anomaly_map.cpu())
                            binary_mask = torch.ones_like(anomaly_map.cpu())
                    elif config['mvtec_path'].split('/')[2] == 'toothbrush': #224
                        if anomaly_map.max() > 35:
                            if anomaly_map.max() > 49:
                                threshold = 40.0
                            else:
                                threshold = 28.0
                            binary_mask = (anomaly_map.cpu() > threshold).float()
                            map_pred = torch.clip(anomaly_map.cpu(), min=anomaly_map.cpu().min(), max=threshold)
                            mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                            mask_pred = mask_pred**2
                        else:
                            mask_pred = torch.ones_like(anomaly_map.cpu())
                            binary_mask = torch.ones_like(anomaly_map.cpu())
                    elif config['mvtec_path'].split('/')[2] == 'hazelnut':
                        if anomaly_map.max() > 55:
                            if anomaly_map.max() > 60:
                                threshold = 46.5
                            elif anomaly_map.max() > 50:
                                threshold = 42.5
                            else:
                                threshold = 37.5
                            binary_mask = (anomaly_map.cpu() > threshold).float()
                            map_pred = torch.clip(anomaly_map.cpu(), min=threshold-anomaly_map.cpu().std(), max=threshold)
                            mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                            mask_pred = mask_pred**2
                        else:
                            mask_pred = torch.ones_like(anomaly_map.cpu())
                            binary_mask = torch.ones_like(anomaly_map.cpu())
                    elif config['mvtec_path'].split('/')[2] == 'cable':
                        if anomaly_map.max() > 40:#112
                            if anomaly_map.max() > 50:#46:
                                threshold = 42.5
                            else:
                                threshold = 36.5
                            binary_mask = (anomaly_map.cpu() > threshold).float()
                            map_pred = torch.clip(anomaly_map.cpu(), min=threshold-anomaly_map.std(), max=threshold)
                            mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                            mask_pred = mask_pred**2
                        else:
                            mask_pred = torch.ones_like(anomaly_map.cpu())
                            binary_mask = torch.ones_like(anomaly_map.cpu())
                    elif config['mvtec_path'].split('/')[2] == 'grid':
                        if anomaly_map.max() > 27:#224
                            if anomaly_map.max() > 40:
                                threshold = 35.0
                            elif anomaly_map.max() > 35.0:
                                threshold = 30.0
                            else:
                                threshold = 26.5#35.0
                            binary_mask = (anomaly_map.cpu() > threshold).float()
                            map_pred = torch.clip(anomaly_map.cpu(), min=anomaly_map.cpu().min(), max=threshold)
                            mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                            mask_pred = mask_pred**2
                        else:
                            mask_pred = torch.ones_like(anomaly_map.cpu())
                            binary_mask = torch.ones_like(anomaly_map.cpu())
                    elif config['mvtec_path'].split('/')[2] == 'zipper':
                        if anomaly_map.max() > 27:#224
                            if anomaly_map.max() > 40:
                                threshold = 35.0
                            elif anomaly_map.max() > 35.0:
                                threshold = 30.0
                            else:
                                threshold = 26.5#35.0
                            binary_mask = (anomaly_map.cpu() > threshold).float()
                            map_pred = torch.clip(anomaly_map.cpu(), min=anomaly_map.cpu().min(), max=threshold)
                            mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                            mask_pred = mask_pred**2
                        else:
                            mask_pred = torch.ones_like(anomaly_map.cpu())
                            binary_mask = torch.ones_like(anomaly_map.cpu())

                    #mask_pred = torch.zeros_like(cls.cpu())
                    #mask_pred[:,:, 40:55, 38:50] = 1.0
                    #mask_pred = transforms.RandomVerticalFlip(p=1.0)(mask_pred)
                    #hr = transforms.RandomVerticalFlip(p=1.0)(hr)
                    #lr = transforms.RandomVerticalFlip(p=1.0)(lr)
                    #binary_mask = mask_pred
                    #mask_pred = cls.cpu()
                    #binary_mask = cls.cpu()
                    #mask_pred = torch.clip(mask_pred, 0., 0.6)
                    
                lst_masks.append(mask_pred.cpu().detach().numpy())

            else:
                mask_pred = None
            if config['return_all_out']:
                print("Return all output: ", config['data'])
                binary_mask = None
                out, x_start_preds, confidences, stats_mean, stats_std = trainer.ema.ema_model.sample(lr, hr, batch_size=lr.shape[0], return_all_timesteps = True, return_all_outputs = config['return_all_out'], mask=binary_mask, ood_confidence_ad = False, min_max_val = min_max_val)
                lst_x_starts.append(np.stack(x_start_preds, axis=1))

                #stats_mean: 1,28,28 stats_std: 1,28,28
                print(np.stack(x_start_preds,axis=1).shape)
                x_starts_sample_avg = np.mean(np.stack(x_start_preds, axis=1)[0], axis=0) # 1,50,1,28,28
                print("After: ", x_starts_sample_avg.shape)
                mean_avg_sample = stats_mean[0]
                std_avg_sample = stats_std[0]

                binary_mask = (cls.cpu()>0).int()#likelihood_implicit_threshold.numpy().astype(int)
                binary_mask = torch.tensor(binary_mask).to(device)
                if config['cond'] == 'IN':
                    binary_mask = 1 - binary_mask
                lr_mask_ood = torch.where(binary_mask > 0, lr, min_max_val[-1])#lr * binary_mask
                lst_lr_mask_ood.append(lr_mask_ood.cpu().detach().numpy())

                lst_confidence.append(np.stack(confidences, axis=1))
                lst_stats_mean.append(stats_mean)
                lst_stats_std.append(stats_std)
            else:
                if config['ood_AD']:
                    if config['branch_out']:
                        #mask = non-binary here
                        print("LR Min {} Max {}, HR Min {} Max {}".format(lr.min(), lr.max(), hr.min(), hr.max()))
                        out = trainer.ema.ema_model.sample(lr, hr, batch_size=lr.shape[0], return_all_timesteps = config['return_all_timesteps'], return_all_outputs = config['return_all_out'], mask = mask_pred.to(device), min_max_val = min_max_val)
                    else:
                        print("LR Min {} Max {}, HR Min {} Max {}".format(lr.min(), lr.max(), hr.min(), hr.max()))
                        out = trainer.ema.ema_model.sample(lr, hr, batch_size=lr.shape[0], return_all_timesteps = config['return_all_timesteps'], return_all_outputs = config['return_all_out'], mask = binary_mask.to(device), min_max_val = min_max_val)
                else:
                    if (config['data'] == 'mnist') and (config['branch_out'] == True) and (config['ood_AD'] == False):
                        mask = torch.zeros_like(lr)
                        length = mask.shape[-1]
                        mask[:,:,:,14:] = 1.0
                        mask = 1.0 - mask
                        #mask =  torch.clip(mask, 0.1, 1.)
                        print("MNIST mask generated")
                    else:
                        if config['mask_x']:
                            cond_img = lr * binary_mask.to(device)
                        mask = None
                    print("LR Min {} Max {}, HR Min {} Max {}".format(lr.min(), lr.max(), hr.min(), hr.max()))
                    out = trainer.ema.ema_model.sample(lr, hr, batch_size=lr.shape[0], return_all_timesteps = config['return_all_timesteps'], return_all_outputs = config['return_all_out'], mask = mask, min_max_val = min_max_val)
            lst.append(torch.nn.MSELoss()(out[:,[-1]].cpu(), hr.cpu()))

            lst_hr.append(hr.cpu().detach().numpy())
            lst_pred.append(out.cpu().detach().numpy())
            lst_lr.append(lr.cpu().detach().numpy())

    #stack all the numpy arrays
    lst = np.array(lst)
    lst_hr = np.concatenate(np.array(lst_hr))
    lst_pred = np.concatenate(np.array(lst_pred))
    lst_lr = np.concatenate(np.array(lst_lr))
    lst_defect_name = np.concatenate(np.array(lst_defect_name))
    np.save(f'hr_all.npy', lst_hr)
    np.save(f'lr_all.npy', lst_lr)
    np.save(f'pred_all.npy', lst_pred)
    np.save(f'defect_name.npy', lst_defect_name)

    if config['return_all_out']:
        lst_x_starts = np.concatenate(np.array(lst_x_starts))
        lst_confidence = np.concatenate(np.array(lst_confidence))
        lst_stats_mean = np.array(lst_stats_mean)
        lst_stats_std = np.array(lst_stats_std)
        np.save(f'confidences.npy', lst_confidence)
        np.save(f'x_starts_pred.npy', lst_x_starts)
        np.save(f'stats_mean.npy', lst_stats_mean)
        np.save(f'stats_std.npy', lst_stats_std)
    if config['ood_AD']:
        print(lst_masks[0].shape)
        lst_masks = np.concatenate(np.array(lst_masks))
        np.save(f'ad_masks.npy', lst_masks)
    ls = np.mean(np.array(lst))
    print("Test los: {:.4f}".format(ls))

    x = np.load('./pred_all.npy')
    y = np.load('./hr_all.npy')
    print(x.shape, x.min(), x.max(), y.shape, y.min(), y.max())
