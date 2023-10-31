from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from datasets.utils.file_utils import get_datasets_user_agent
import io
import urllib
from torch.utils.data.dataloader import default_collate
import nibabel as nib
import glob

from medpy.io import load
from medpy.io import header

import idx2numpy
from torchvision import transforms as T
import torchvision.transforms as transforms

torch.set_warn_always(False)

USER_AGENT = get_datasets_user_agent()

# helpers functions

def exists(val):
    return val is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# dataset, dataloader, collator

def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))

    if batch == []:
        return None
    
    return default_collate(batch)

class supervisedIQT(Dataset):
    def __init__(self, config, lr_files, hr_files, train=True):
        self.config = config
        self.lr_files = lr_files
        self.hr_files = hr_files

        self.mean_lr = self.config['Data']['mean']#202.68109075616067 #35.493949511348724
        self.std_lr = self.config['Data']['std']#346.51374798642223 #37.11344433531084

        if self.config['Train']['batch_sample']:
            self.patch_size = self.config['Train']['patch_size_sub'] * self.config['Train']['batch_sample_factor']
        else:
            self.patch_size = self.config['Train']['patch_size_sub']
        self.train = train
        if self.train:
            self.ratio = 0.2
        else:
            self.ratio = 0.8

        self.files_lr = []
        self.files_hr = []
        
        for i in range(len(self.lr_files)):
            self.files_lr.append(self.lr_files[i])
            self.files_hr.append(self.hr_files[i])

    def __len__(self):
        return len(self.files_lr)

    def normalize(self, img, mode='lr'): # transform 3D array to tensor

        if self.config['Data']['norm'] == 'min-max':
            return 2*(((img-img.min())/(img.max()-img.min()))-0.5)
        return (img - self.mean_lr)/self.std_lr
    
    def cube(self,data):

        hyp_norm = data

        if len(hyp_norm.shape)>3:
            hyp_norm = hyp_norm[:,:, 2:258, 27:283]
        else:
            hyp_norm = hyp_norm[2:258, 27:283, :256]

        return hyp_norm

    def __getitem__(self, idx):
        
        self.lr = self.files_lr[idx]
        self.hr = self.lr.replace('lr_norm',self.config['Data']['groundtruth_fname'])#'T1w_acpc_dc_restore_brain')   #'T1w_acpc_dc_restore_brain_sim036T_4x_groundtruth_norm')
        
        self.lr = nib.load(self.lr)
        self.lr_affine = self.lr.affine
        self.lr = torch.tensor(self.lr.get_fdata().astype(np.float32))
        self.img_shape = self.lr.shape
        self.hr = nib.load(self.hr)
        self.hr_affine = self.hr.affine
        self.hr = torch.tensor(self.hr.get_fdata().astype(np.float32))
        
        #Cube
        low, high = 0, 256 #16, 240 
        
        assert self.lr.shape == (256,256,256), f'lr must be 256 256 256 but got {self.lr.shape}'
        assert self.hr.shape == (256,256,256), f'hr must be 256 256 256 but got {self.hr.shape}'

        self.lr = self.lr[low:high,low:high,low:high]
        self.hr = self.hr[low:high,low:high,low:high] 

        random_idx = np.random.randint(low=0, high=(high-low)-self.patch_size, size=3)
        self.lr = self.lr[random_idx[0]:random_idx[0]+self.patch_size, random_idx[1]:random_idx[1]+self.patch_size, random_idx[2]:random_idx[2]+self.patch_size]
        self.hr = self.hr[random_idx[0]:random_idx[0]+self.patch_size, random_idx[1]:random_idx[1]+self.patch_size, random_idx[2]:random_idx[2]+self.patch_size]
	
        non_zero = np.count_nonzero(self.lr)
        self.total_voxel = self.patch_size * self.patch_size * self.patch_size
        non_zero_proportion = (non_zero/self.total_voxel)
        if (non_zero_proportion < self.ratio):
            return self.__getitem__(idx)
                            
        self.lr = self.normalize(self.lr, mode='lr')
        self.hr = self.normalize(self.hr, mode='hr')
            
        sample_lr = torch.unsqueeze(self.lr, 0)
        sample_hr = torch.unsqueeze(self.hr, 0)
        
        if self.train:
            return sample_hr, sample_lr
        else:    
            return sample_hr, sample_lr

class IQTDataset(Dataset):
    def __init__(
        self,
        hr_files,
        lr_files,
    ):
        self.hrfiles = glob.glob(hr_files)
        self.lrfiles = glob.glob(lr_files)
        self.mid = 128
        
        assert len(self.hrfiles) == len(self.hrfiles), "Length should be same"
    
    def transform(self, img, size=(256,256)):
        return TF.resize(img, size)
        
    def normalize(self, img):
        img = (img-img.min())/(img.max()-img.min())
        return img
    
    def np2tensor(self, x):
        x = torch.tensor(x)
        x = torch.unsqueeze(x,0)
        return x

    def __len__(self):
        return len(self.hrfiles)

    def __getitem__(self, idx):

        hrfile = self.hrfiles[idx]
        lrfile = self.hrfiles[idx].replace('T1w_acpc_dc_restore_brain_sim036T_4x_groundtruth_norm.nii', 'lr_norm.nii')
        idx = self.mid
        
        hrimg = nib.load(hrfile).get_fdata().astype(np.float32)[:,idx,:]
        hrimg = self.np2tensor(hrimg)
        hrimg = self.transform(hrimg)
        hrimg = self.normalize(hrimg)
        
        lrimg = nib.load(lrfile).get_fdata().astype(np.float32)[:,idx,:]
        lrimg = self.np2tensor(lrimg)
        lrimg = self.transform(lrimg)
        lrimg = self.normalize(lrimg)
  
        return hrimg, lrimg
    

#IMAGE TRANSLATION for BRATS
class MedDataset(Dataset):
    def __init__(
        self,
        config,
        mri_files,
        train=True,
        tumor=False
    ):
        self.config = config
        self.mris = mri_files
        self.train = train
        self.tumor = tumor
        self.lst = []
        for mri in self.mris:
            t1 = glob.glob(mri + '/VSD.Brain.XX.O.MR_T1/*.mha')[0]
            flair = glob.glob(mri + '/VSD.Brain.XX.O.MR_Flair/*.mha')[0]
            seg = glob.glob(mri + 'VSD.Brain_*more.XX*/*.mha')[0]

            seg, _ = load(seg)
            self.cnt = 0
            #only healthy subjects
            if self.train:
                for i in range(50,140,2):
                    slice = seg[:,:,i]
                    if np.unique(slice).size == 1:
                        self.lst.append([t1,flair,slice,i])
            else:
                if self.tumor:
                    low,high,skip = 80,100,5
                    for i in range(low, high, skip):
                        slice = seg[:,:,i]
                        if np.unique(slice).size != 1:
                            self.lst.append([t1,flair,slice,i])
                            self.cnt+=1
                        if self.cnt == 2:
                            break
                else:
                    low,high,skip=60,120,5
                    for i in range(low,high,skip):
                        slice = seg[:,:,i]
                        if np.unique(slice).size == 1:
                            self.lst.append([t1,flair,slice,i])
                            self.cnt+=1
                        if self.cnt == 2:
                            break
    def transform(self, img, size=(224,224)):
        return T.CenterCrop(size)(img)
        
    def normalize(self, img, mode='t1'):
        if mode == 't1':
            img = (img - self.config['mean_t1'])/(self.config['std_t1'])
        else:
            img = (img - self.config['mean_flair'])/(self.config['std_flair'])
        #img = (img-img.min())/(img.max()-img.min())
        return img
    
    def np2tensor(self, x):
        x = torch.tensor(x)
        x = torch.unsqueeze(x,0)
        return x

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        t1, flair, seg, slice = self.lst[idx][0], self.lst[idx][1], self.lst[idx][2], self.lst[idx][3]
        t1_sample, _ = load(t1)
        t1_sample = t1_sample[:,:,slice].astype(np.float32)
        flair_sample, _ = load(flair)
        flair_sample = flair_sample[:,:,slice].astype(np.float32)
        seg = torch.tensor(seg.astype(np.float32))

        t1_sample = self.np2tensor(t1_sample)
        t1_sample = self.transform(t1_sample)
        t1_sample = self.normalize(t1_sample, mode='t1')
        
        flair_sample = self.np2tensor(flair_sample)
        flair_sample = self.transform(flair_sample)
        flair_sample = self.normalize(flair_sample, mode='flair')

        seg = self.np2tensor(seg)
        seg = self.transform(seg)
  
        return flair_sample, t1_sample, seg #flair_sample, t1_sample, seg
 
    #IMAGE TRANSLATION for BRATS
class SingleMedDataset(Dataset):
    def __init__(
        self,
        config,
        mri_file,
    ):
        self.config = config
        self.mri = mri_file
        self.lst = []

        t1 = glob.glob(self.mri + '/VSD.Brain.XX.O.MR_T1/*.mha')[0]
        flair = glob.glob(self.mri + '/VSD.Brain.XX.O.MR_Flair/*.mha')[0]
        seg = glob.glob(self.mri + 'VSD.Brain_*more.XX*/*.mha')[0] #'/VSD.Brain_1more.XX.O.OT/*.mha')[0]

        seg, _ = load(seg)
        self.cnt = 0

        low,high=0,seg.shape[-1]
        for i in range(low,high):
            slice = seg[:,:,i]
            self.lst.append([t1,flair,slice,i])

    def transform(self, img, size=(224,224)):
        return T.CenterCrop(size)(img)
        
    def normalize(self, img, mode='t1'):
        if mode == 't1':
            img = (img - self.config['mean_t1'])/(self.config['std_t1'])
        else:
            img = (img - self.config['mean_flair'])/(self.config['std_flair'])
        #img = (img-img.min())/(img.max()-img.min())
        return img
    
    def np2tensor(self, x):
        x = torch.tensor(x)
        x = torch.unsqueeze(x,0)
        return x

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        t1, flair, seg, slice = self.lst[idx][0], self.lst[idx][1], self.lst[idx][2], self.lst[idx][3]
        t1_sample, _ = load(t1)
        t1_sample = t1_sample[:,:,slice].astype(np.float32)
        flair_sample, _ = load(flair)
        flair_sample = flair_sample[:,:,slice].astype(np.float32)
        seg = torch.tensor(seg.astype(np.float32))

        t1_sample = self.np2tensor(t1_sample)
        t1_sample = self.transform(t1_sample)
        t1_sample = self.normalize(t1_sample, mode='t1')
        
        flair_sample = self.np2tensor(flair_sample)
        flair_sample = self.transform(flair_sample)
        flair_sample = self.normalize(flair_sample, mode='flair')
  
        return flair_sample, t1_sample, seg

# #MNIST
class MNIST(Dataset):
    def __init__(
        self,
        config,
        files, 
        labels,
        train=True,
        num=8,
        max_file=None
    ):
        self.config = config
        self.data = files #idx2numpy.convert_from_file(self.config['mnist_path'])
        self.labels = labels #idx2numpy.convert_from_file(self.config['mnist_labels_path'])
        self.num = num
        self.train = train
        self.mean = 0
        self.std = 0.01
        self.lst = []

        if max_file is not None:
            self.cnt = max_file
        else:
            self.cnt = None

        for i in range(len(self.data)):
            if self.labels[i] == self.num:
                self.lst.append([self.data[i], self.labels[i]])
            if self.cnt:
                if self.cnt == len(self.lst):
                    break
                            
    def transform(self, img, size=(24, 24)):
        #create transform for MNIST
        if self.train:
            T = transforms.Compose([
                #convert torch tensor to PIL image
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ])
        else:
            T = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
        return T(img)
    
    def np2tensor(self, x):
        x = torch.tensor(x)
        return x
    
    def add_noise(self, x):
        x = x + torch.randn(x.size()) * self.std + self.mean
        return x
    
    def normalize(self, x):
        return (x - x.min())/(x.max() - x.min())
    
    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        img, label = self.lst[idx][0], self.lst[idx][1]
        img = self.np2tensor(img)
        label = self.np2tensor(label)
        img = self.transform(img)
        #downsample img
        img_down = img[:,::2,::2] #28x28 -> 10x10
        #img_down = self.add_noise(img_down)
        #upsample img
        img_down = img_down.unsqueeze(0)
        img_down = F.interpolate(img_down, size=(img.shape[-1], img.shape[-1]), mode='bilinear', align_corners=False)
        img_down = self.add_noise(img_down)
        #if self.train:
        #img_down = self.add_noise(img_down)
        #img_down = self.normalize(img_down)

        if len(img.shape) == 4:
            img = img.squeeze(0)
        if len(img_down.shape) == 4:
            img_down = img_down.squeeze(0)
  
        return img, img_down, label
