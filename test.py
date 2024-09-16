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
from patchcore import PatchcoreModel
import timm
from unet_model import UNet
import time

def set_min_max_val(config, mode = 'mri'):
    if mode == 'mri':
        if not config['translate_zero']:
            max_val = (4096-config['mean_flair'])/config['std_flair']
            min_val = (0-config['mean_flair'])/config['std_flair']
            min_val_t1 = (0-config['mean_t1'])/config['std_t1']
        else:
            min_val2 = (0-config['mean_flair'])/config['std_flair'] 
            min_val = 0.
            max_val = (4096-config['mean_flair'])/config['std_flair']
            max_val = max_val + torch.abs(torch.tensor(min_val2))
            min_val_t1 = 0.
        return max_val, min_val, min_val_t1
    elif mode == 'mnist':
        min_val = 0.0
        max_val = 2.0
        return max_val, min_val
    elif mode == 'mvtec':
        min_val = 0.0
        max_val = 2.0
        return max_val, min_val

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
    
    # dataset and dataloader for MNIST
    file_path = '/cluster/project0/IQT_Nigeria/skim/diffusion_az/test/MNIST/raw/t10k-images-idx3-ubyte'
    label_path = '/cluster/project0/IQT_Nigeria/skim/diffusion_az/test/MNIST/raw/t10k-labels-idx1-ubyte'
    
    mri_files = idx2numpy.convert_from_file(file_path)
    mri_labels = idx2numpy.convert_from_file(label_path)
    
    #For testing on OOD data
    if config['ood']:
        if config['data'] == 'mnist':
            ds_test = MNIST(config, mri_files, mri_labels, train=False, num=config['anomaly_name'], max_file=100) 
        elif config['data'] == 'mri':
            print("Data: Brain")
            np.random.seed(42)
            mri_files = np.array(glob.glob(config['mri_files']))
            np.random.shuffle(mri_files)

            #split mri_files into train and test
            train_split = int(0.5 * len(mri_files))
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
            
            ds_test = MvtecDatasetSR(mri_files, train=False, mode=config['anomaly_name'])
           
    #For testing on IND data
    else:
        if config['data'] == 'mnist':
            ds_test = MNIST(config, mri_files, mri_labels, train=False, num=[8], max_file=100) 
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
            ds_test = MvtecDatasetSR(mri_files, train=True)

    dl_test = DataLoader(ds_test, batch_size = 1, shuffle = False, pin_memory = True, num_workers = cpu_count())

    data = next(iter(dl_test))
    print(len(ds_test), data[0].shape, data[1].shape)

    #Initialize U-Net model 
    if config['data'] == 'mri': 
        max_val, min_val, min_val_t1 = set_min_max_val(config, mode = 'mri')
        min_max_val = (min_val, max_val, min_val_t1) 
        model = Unet(dim=config['dim'], init_dim=config['dim'], mode=config['data'])
    elif config['data'] == 'mnist':
        max_val, min_val = set_min_max_val(config, mode = 'mnist')
        min_max_val = (min_val, max_val) 
        model = Unet(dim=config['dim'], init_dim=config['dim'], dim_mults = (1, 2, 4), full_attn = (False, False, True), mode=config['data'])
    elif ('mvtec' in config['data']):
        max_val, min_val = set_min_max_val(config, mode = 'mvtec')
        min_max_val = (min_val, max_val)
        channels, out_dim = 3, 3
        if config['data'] == 'mvtecSR':
            model = Unet(dim=config['dim'], init_dim=config['dim'], dim_mults = (1, 2, 4), full_attn = (False, False, True), channels = channels, out_dim=out_dim, mode=config['data'])
        else:
            model = Unet(dim=config['dim'], init_dim=config['dim'], channels = channels, out_dim=out_dim, mode=config['data'])

    #Calculate number of parameters    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", pytorch_total_params)

    #Load the model
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

    #if you want to segment OOD from the conditional image
    if config['ood_AD']:
        
        #Initialize Patchcore model
        MODEL = "patchcore"  # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
        CONFIG_PATH = '/cluster/project0/IQT_Nigeria/skim/diffusion_az/test/anomalib/' + f"src/anomalib/models/{MODEL}/config.yaml"
        # pass the config file to model, callbacks and datamodule
        config_ad = get_configurable_parameters(config_path=CONFIG_PATH)

        backbone = 'wide_resnet50_2'
        if 'resnet' in backbone:
            layers = ['layer2', 'layer3']
        else:
            layers = [2, 3]
            
        if config['data'] == 'mnist':
            img_size = 84
        else:
            img_size = 224
        patchcore = PatchcoreModel(input_size = [img_size, img_size], layers = layers,backbone= backbone, pre_trained= True, num_neighbors= 9)
        if config['data'] == 'mnist':
            pretrained = np.load(f'/cluster/project0/IQT_Nigeria/skim/diffusion_az/test/memory_bank_mnist_train.npy')
        elif 'mvtec' in config['data']: 
            pretrained = np.load('/home/seunghki/mnist_az/memory_bank_mvtec_{}.npy'.format(config['mvtec_path'].split('/')[5]))
        else:
            pretrained = np.load(f'/home/seunghki/mnist_az/memory_bank_mri_t12flair.npy')
        patchcore.memory_bank = torch.from_numpy(pretrained).to(device)
        patchcore.training = False
        patchcore.feature_extractor = patchcore.feature_extractor.to(device)
        patchcore.feature_extractor.eval()

    #Test the model
    with torch.inference_mode():
        lst = []
        lst_hr = []
        lst_pred = []
        lst_lr = []
        lst_masks = []
        lst_defect_name = []
        times = []
        print("Test start!")
        for i, data in enumerate(dl_test):
            if len(data) == 3:
                hr, lr, cls = data
                hr, lr, cls = hr.to(device), lr.to(device), cls.to(device)
            elif len(data) == 4:
                hr, lr, cls, defect = data
                hr, lr, cls = hr.to(device), lr.to(device), cls.to(device)
                lst_defect_name.append(defect)

            if config['ood_AD']:
                print("Segmenting OOD from conditional image...")
                if lr.shape[1] != 3: #the input image to PatchCore must be 3 channels
                    lr_ad = lr.repeat(1, 3, 1, 1)
                else:
                    lr_ad = lr.clone()

                #denormalize the images then normalize them again for PatchCore
                if config['data'] == 'mri':
                    #denormalize first
                    if config['translate_zero']:
                        mini = (0-config['mean_t1'])/config['std_t1']
                        lr_ad = lr_ad - torch.abs(torch.tensor(mini))
                    lr_ad = lr_ad[:,0]*config['std_t1'] + config['mean_t1'] #denormalize the original image
                    
                    #another method is to segment using pre-trained segmentation model instead of patchcore
                    if config['ood_detector']['seg']:
                        lr_ad = lr - torch.abs(torch.tensor(mini))
                        print(lr_ad.min(), lr_ad.max())
                        seg_model = UNet()
                        seg_model.load_state_dict(torch.load(config['ood_detector']['seg_model']))
                        seg_model = seg_model.to(device)
                        seg_model.eval()
                        sigmoid = nn.Sigmoid()
                    else:  
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
                
                #segment the OOD from the conditional image using patchcore
                if not config['ood_detector']['seg']:
                    lr_ad = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(lr_ad)

                    pred_anomalymap = patchcore(lr_ad.to(device))
                    anomaly_map, pred_score = pred_anomalymap["anomaly_map"], pred_anomalymap["pred_score"]
                    if ('mvtec' in config['data']) or (config['data'] == 'mnist'):
                        anomaly_map = F.interpolate(anomaly_map, size=(config['img_size'], config['img_size']), mode='bilinear', align_corners=False)
                    print("Anomaly score: ", anomaly_map.max())
                    
                #separate the OOD from the conditional image
                if config['data'] == 'mnist':
                    if config['mnist_cls'] == '8to3':
                        if anomaly_map.max() > 37.0:
                            if anomaly_map.max() > 44:
                                threshold = 41.7
                            elif anomaly_map.max() > 40.0:
                                threshold = 38.2
                            else:
                                threshold = 35.0
                            binary_mask = (anomaly_map.cpu() > threshold).float()
                            map_pred = torch.clip(anomaly_map.cpu(), min=threshold-anomaly_map.std(), max=threshold)
                            mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                            mask_pred = mask_pred **2
                        else:
                            mask_pred = torch.ones_like(anomaly_map.cpu())
                            binary_mask = torch.ones_like(anomaly_map.cpu())
                    elif config['mnist_cls'] == '8to5':
                        if anomaly_map.max() > 58.5:
                            if anomaly_map.max() > 71.0:
                                threshold = 61.0
                            elif anomaly_map.max() > 65:
                                threshold = 57.0
                            else:
                                threshold = 55.0
                            binary_mask = (anomaly_map.cpu() > threshold).float()
                            map_pred = torch.clip(anomaly_map.cpu(), min=threshold-anomaly_map.std(), max=threshold)
                            mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                            mask_pred = mask_pred **2
                        else:
                            mask_pred = torch.ones_like(anomaly_map.cpu())
                            binary_mask = torch.ones_like(anomaly_map.cpu())
                if config['data'] == 'mri':
                        
                    if config['ood_detector']['seg']:
                        print("Segmentation")
                        seg_out = seg_model(lr_ad)
                        mask_pred = sigmoid(seg_out).detach().cpu()
                        binary_mask = (mask_pred > 0.5).float()
                        mask_pred = binary_mask
                    else:
                        if 't12flair' in config['ProjectName']:
                            if anomaly_map.max() > 43:
                                if anomaly_map.max() > 60:
                                    threshold = anomaly_map.max()-12
                                elif anomaly_map.max() > 51:
                                    threshold = 47
                                elif anomaly_map.max() > 48.5:
                                    threshold = 44
                                else:
                                    threshold = 42
                                binary_mask = (anomaly_map.cpu() > threshold).float()
                                map_pred = torch.clip(anomaly_map.cpu(), min=threshold-anomaly_map.std(), max=threshold)
                                mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                                mask_pred = mask_pred **2
                            else:
                                mask_pred = torch.ones_like(anomaly_map.cpu())
                                binary_mask = torch.ones_like(anomaly_map.cpu())

                        elif 'flair2t1' in config['ProjectName']:
                            if anomaly_map.max() > 43:
                                if anomaly_map.max() > 60:
                                    threshold = 47
                                elif anomaly_map.max() > 50:
                                    threshold = 43
                                else:
                                    threshold = 42
                                binary_mask = (anomaly_map.cpu() > threshold).float()
                                map_pred = torch.clip(anomaly_map.cpu(), min=threshold-anomaly_map.std(), max=threshold)
                                mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                                mask_pred = mask_pred **2
                            else:
                                mask_pred = torch.ones_like(anomaly_map.cpu())
                                binary_mask = torch.ones_like(anomaly_map.cpu())

                    cls[cls>0.0] = 1.0
                    # mask_pred = cls.cpu()
                    # binary_mask = cls.cpu()

                if 'mvtec' in config['data']:
                    if config['mvtec_path'].split('/')[5] == 'transistor': #64
                        print("TRANSISTOR!!")
                        if anomaly_map.max() > 32:
                            if anomaly_map.max() > 40.0:
                                threshold = 33.5
                            elif anomaly_map.max() > 36.8:
                                threshold = anomaly_map.max() - 2*anomaly_map.cpu().std()
                            elif anomaly_map.max() > 35.0:
                                threshold = anomaly_map.max() - 1*anomaly_map.cpu().std()
                            else:
                                threshold = 29.5
                            binary_mask = (anomaly_map.cpu() > threshold).float()
                            map_pred = torch.clip(anomaly_map.cpu(), min=threshold-0.5*anomaly_map.cpu().std(), max=threshold)
                            mask_pred = (map_pred - map_pred.min()) / (threshold - map_pred.min())
                            mask_pred = mask_pred**2
                        else:
                            mask_pred = torch.ones_like(anomaly_map.cpu())
                            binary_mask = torch.ones_like(anomaly_map.cpu())
                    elif config['mvtec_path'].split('/')[5] == 'toothbrush': #224
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
                    elif config['mvtec_path'].split('/')[5] == 'grid':
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

                #### Or you can either manually separate the OOD from the conditional image just like in the motivational exp.1 ####
                #### Uncomment the code below ####
                mask_pred = torch.zeros_like(anomaly_map.cpu())
                mask_pred[:,:, :, :7] = 1.0
                binary_mask = mask_pred

                lst_masks.append(mask_pred.cpu().detach().numpy())

            else:
                mask_pred = None
            
            if config['ood_AD']:
                if config['branch_out']:
                    #mask = non-binary here
                    print("LR Min {} Max {}, HR Min {} Max {}".format(lr.min(), lr.max(), hr.min(), hr.max()))
                    start = time.time()
                    out = trainer.ema.ema_model.sample(lr, hr, batch_size=lr.shape[0], return_all_timesteps = config['return_all_timesteps'], return_all_outputs = config['return_all_out'], mask = mask_pred.to(device), min_max_val = min_max_val)
                    times.append(time.time()-start) 
                else:
                    start = time.time()
                    print("LR Min {} Max {}, HR Min {} Max {}".format(lr.min(), lr.max(), hr.min(), hr.max()))
                    out = trainer.ema.ema_model.sample(lr, hr, batch_size=lr.shape[0], return_all_timesteps = config['return_all_timesteps'], return_all_outputs = config['return_all_out'], mask = binary_mask.to(device), min_max_val = min_max_val)
                    times.append(time.time()-start) 
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
                start = time.time()
                print("LR Min {} Max {}, HR Min {} Max {}".format(lr.min(), lr.max(), hr.min(), hr.max()))
                out = trainer.ema.ema_model.sample(lr, hr, batch_size=lr.shape[0], return_all_timesteps = config['return_all_timesteps'], return_all_outputs = config['return_all_out'], mask = mask, min_max_val = min_max_val)
                times.append(time.time()-start) 
            lst.append(torch.nn.MSELoss()(out[:,[-1]].cpu(), hr.cpu()))

            lst_hr.append(hr.cpu().detach().numpy())
            lst_pred.append(out.cpu().detach().numpy())
            lst_lr.append(lr.cpu().detach().numpy())

    #stack all the numpy arrays
    lst = np.array(lst)
    lst_hr = np.concatenate(np.array(lst_hr))
    lst_pred = np.concatenate(np.array(lst_pred))
    lst_lr = np.concatenate(np.array(lst_lr))
    if len(times) > 1:
        times = np.array(times)
    np.save(f'hr_all.npy', lst_hr)
    np.save(f'lr_all.npy', lst_lr)
    if config['ood_AD']:
        np.save(f'pred_all.npy', lst_pred) #_localdiff_{config["oct_ad_path"].split("/")[-2]}2.npy', lst_pred)
    else:
        np.save(f'pred_all.npy', lst_pred)
    if config['data'] == 'mvtec':
        lst_defect_name = np.concatenate(np.array(lst_defect_name))
        np.save(f'defect_name.npy', lst_defect_name)

    if config['ood_AD']:
        print(lst_masks[0].shape)
        lst_masks = np.concatenate(np.array(lst_masks))
        np.save(f'ad_masks.npy', lst_masks)
    ls = np.mean(np.array(lst))
    print("Test loss: {:.4f}".format(ls))
    print("Average sampling time: {:.4f}".format(np.mean(times)))
