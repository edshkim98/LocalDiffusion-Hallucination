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
import torchvision.transforms as transforms

from datasets.utils.file_utils import get_datasets_user_agent
import io
import urllib
from torch.utils.data.dataloader import default_collate
import nibabel as nib
import glob

from medpy.io import load
from medpy.io import header

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
class MvtecDataset(Dataset):
    def __init__(self, files, train=False, mode=None):
        self.train = train
        self.files = files
        self.mode = mode
        self.lst = []
        for file in self.files:
            if self.train:
                if 'good' in file:
                    self.lst.append(file)
            else:
                if self.mode == None:
                    if 'good' not in file:
                        self.lst.append(file)
                else:
                    if self.mode in file:
                        self.lst.append(file)
    def __len__(self):
        return len(self.lst)
    
    def RGB2Gray(self, x):
        x = torch.matmul(x[:3,...].permute(1,2,0), torch.tensor([0.2989, 0.5870, 0.1140]))
        return x
    
    def transform(self, img, size=(224, 224)):
        T = transforms.Compose([
            transforms.Resize(size),
            #transforms.ColorJitter(brightness=0.5, hue=0.5),
            transforms.ToTensor()
        ])
        img = T(img)
        return img
    
    def __getitem__(self, idx):
        img_rgb = Image.open(self.lst[idx]).convert('RGB')
        img_rgb = self.transform(img_rgb)
        img_gray = self.RGB2Gray(img_rgb)
        if len(img_gray.shape) == 2:
            img_gray = img_gray.unsqueeze(0)
        label = 0 if 'good' in self.lst[idx] else 1

        return img_rgb, img_gray, label

class MvtecDatasetGray(Dataset):
    def __init__(self, files, train=False, mode=None, max_num=False, denoise=False):
        self.train = train
        self.denoise = denoise
        self.files = files
        self.mode = mode
        self.max_num = max_num
        self.lst = []
        for file in self.files:
            if self.train:
                if 'good' in file:
                    self.lst.append(file)
                if self.max_num is not False:
                    if len(self.lst) == self.max_num:
                        break
            else:
                if self.mode == None:
                    if 'good' not in file:
                        self.lst.append(file)
                else:
                    if self.mode in file:
                        self.lst.append(file)
                    if len(self.lst) == self.max_num:
                        break
    def __len__(self):
        return len(self.lst)
    
    def RGB2Gray(self, x):
        x = torch.matmul(x[:3,...].permute(1,2,0), torch.tensor([0.2989, 0.5870, 0.1140]))
        return x
    
    def transform(self, img, size=(112, 112)):
        T = transforms.Compose([
            transforms.Resize(size),
            #transforms.ColorJitter(brightness=0.5, hue=0.5),
            transforms.ToTensor()
        ])
        img = T(img)
        return img
    def salt_and_pepper_noise(self, image, salt_pepper_ratio=0.5, amount=0.05):
        """
        Add salt and pepper noise to a grayscale image tensor.

        :param image: A grayscale image tensor of shape (H, W) or (1, H, W).
        :param salt_pepper_ratio: The ratio of salt to pepper noise.
        :param amount: The proportion of image pixels to alter with noise.
        :return: Noisy image tensor.
        """

        # Create a copy of the input image to avoid modifying the original one
        noisy_image = image.clone()

        # Calculate the number of pixels to alter based on the specified amount
        num_salt = int(round(amount * image.numel() * salt_pepper_ratio))
        num_pepper = int(round(amount * image.numel() * (1.0 - salt_pepper_ratio)))

        # Add Salt noise (set pixels to 1)
        indices = torch.randperm(image.numel())[:num_salt]
        noisy_image.view(-1)[indices] = 1

        # Add Pepper noise (set pixels to 0)
        indices = torch.randperm(image.numel())[:num_pepper]
        noisy_image.view(-1)[indices] = 0

        return noisy_image
    
    def __getitem__(self, idx):
        img_rgb = Image.open(self.lst[idx]).convert('RGB')
        img_rgb = self.transform(img_rgb)
        
        img_gray = self.RGB2Gray(img_rgb)
        if self.denoise:
            img_gray_down = self.salt_and_pepper_noise(img_gray)
            img_gray = img_gray * 2
            img_gray_down = img_gray_down * 2
        else:
            img_gray = img_gray * 2
            img_size = img_gray.shape[-1] // 2
            img_gray_down = img_gray.unsqueeze(0).unsqueeze(0)
            img_gray_down = F.interpolate(img_gray_down, scale_factor=0.5, mode='nearest')#img_rgb[:,::2,::2]
            img_gray_down = F.interpolate(img_gray_down, scale_factor=2.0, mode='bilinear', align_corners=False)
            #img_gray = img_gray * 2
            #img_gray_down = img_gray[::2,::2]
            #img_gray_down = img_gray_down.unsqueeze(0).unsqueeze(0)
            #img_gray_down = F.interpolate(img_gray_down, size=(img_gray.shape[-1], img_gray.shape[-1]), mode='bilinear', align_corners=False)

        if len(img_gray_down.shape) == 4:
            img_gray_down = img_gray_down.squeeze(0)
        if len(img_gray.shape) == 2:
            img_gray = img_gray.unsqueeze(0)
        if len(img_gray_down.shape) == 2:
            img_gray_down = img_gray_down.unsqueeze(0)

        if not self.train:
            label = self.lst[idx].replace('test', 'ground_truth')
            name = label.split('/')[-1][:-4] + '_mask.png'
            label = label.replace(label.split('/')[-1], name)
            label = self.transform(Image.open(label))
            label[label > 0] = 1
        else:
            label = 0 if 'good' in self.lst[idx] else 1

        return img_gray, img_gray_down, label

class MvtecDatasetSR(Dataset):
    def __init__(self, files, train=False, mode=None, max_num=False, mask_train=False, denoise=False):
        self.train = train
        self.mask_train = mask_train
        self.files = files
        self.mode = mode
        self.max_num = max_num
        self.denoise = denoise
        self.lst = []
        for file in self.files:
            if self.train:
                if 'good' in file:
                    self.lst.append(file)
                if self.max_num is not False:
                    if len(self.lst) == self.max_num:
                        break
            else:
                if self.mode == None:
                    if 'good' not in file:
                        self.lst.append(file)
                else:
                    if self.mode in file:
                        self.lst.append(file)
                    if len(self.lst) == self.max_num:
                        break
    def __len__(self):
        return len(self.lst)
    
    def RGB2Gray(self, x):
        x = torch.matmul(x[:3,...].permute(1,2,0), torch.tensor([0.2989, 0.5870, 0.1140]))
        return x
    
    def select_patch(self, img, img_down):
        img_new = torch.zeros_like(img)
        img_down_new = torch.zeros_like(img_down)
        mask = torch.zeros_like(img)
        size = np.random.randint(img.shape[-1]/4, img.shape[-1]/2,2)
        x = np.random.randint(0, img.shape[-1]-size[0]-1)
        y = np.random.randint(0, img.shape[-1]-size[1]-1)
        img_new[:,x:x+size[0],y:y+size[1]] = img[:,x:x+size[0],y:y+size[1]]
        img_down_new[:,x:x+size[0],y:y+size[1]] = img_down[:,x:x+size[0],y:y+size[1]]
        mask[:,x:x+size[0],y:y+size[1]] = 1

        return img_new, img_down_new, mask
    
    def salt_and_pepper_noise(self, image, salt_pepper_ratio=0.5, amount=0.02):
        """
        Add salt and pepper noise to a grayscale image tensor.

        :param image: A grayscale image tensor of shape (H, W) or (1, H, W).
        :param salt_pepper_ratio: The ratio of salt to pepper noise.
        :param amount: The proportion of image pixels to alter with noise.
        :return: Noisy image tensor.
        """
        if not self.train:
            torch.manual_seed(0)
            torch.random.manual_seed(0)

        # Create a copy of the input image to avoid modifying the original one
        noisy_image = image.clone()

        # Calculate the number of pixels to alter
        num_pixels = int(amount * image.numel() / 3)  # Dividing by 3 for three channels
        num_salt = int(round(num_pixels * salt_pepper_ratio))
        num_pepper = num_pixels - num_salt

        # Generate random indices for salt noise
        salt_indices = torch.randperm(image.nelement() // 3)[:num_salt]
        noisy_image.view(3, -1)[:,salt_indices] = torch.tensor([1.0, 1.0, 1.0]).to(torch.float32).unsqueeze(1).repeat(1, num_salt)

        # Generate random indices for pepper noise
        pepper_indices = torch.randperm(image.nelement() // 3)[:num_pepper]
        noisy_image.view(3, -1)[:,pepper_indices] = torch.tensor([0.0, 0.0, 0.0]).to(torch.float32).unsqueeze(1).repeat(1, num_pepper)

        return noisy_image
       
    def transform(self, img, size=(112, 112)):
        T = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])
        img = T(img)
        return img
    
    def __getitem__(self, idx):
        img_rgb = Image.open(self.lst[idx]).convert('RGB')
        img_rgb = self.transform(img_rgb)

        if self.denoise:
            img_rgb_down = self.salt_and_pepper_noise(img_rgb)
            img_rgb = img_rgb * 2
            img_rgb_down = img_rgb_down * 2
        else:
            img_rgb = img_rgb * 2
            img_size = img_rgb.shape[-1] // 2
            img_rgb_down = img_rgb.unsqueeze(0)
            img_rgb_down = F.interpolate(img_rgb_down, size=(img_size, img_size), mode='nearest')#img_rgb[:,::2,::2]
            img_rgb_down = F.interpolate(img_rgb_down, size=(img_rgb.shape[-1], img_rgb.shape[-1]), mode='bilinear', align_corners=False)

            #img_rgb_down = img_rgb[:,::2,::2]
            #img_rgb_down = img_rgb_down.unsqueeze(0)
            #img_rgb_down = F.interpolate(img_rgb_down, size=(img_rgb.shape[-1], img_rgb.shape[-1]), mode='bilinear', align_corners=False)
        if len(img_rgb_down.shape) == 4:
            img_rgb_down = img_rgb_down.squeeze(0)

        if self.mask_train:
            img_rgb, img_rgb_down, mask = self.select_patch(img_rgb, img_rgb_down)
            return img_rgb, img_rgb_down, mask
        
        if not self.train:
            label = self.lst[idx].replace('test', 'ground_truth')
            name = label.split('/')[-1][:-4] + '_mask.png'
            label = label.replace(label.split('/')[-1], name)
            label = self.transform(Image.open(label))
            label[label > 0] = 1
        else:
            label = 0 if 'good' in self.lst[idx] else 1

        return img_rgb, img_rgb_down, label


#IMAGE TRANSLATION for BRATS
class MedDataset_png(Dataset):
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
            t1 = mri.replace('flair','t1')
            seg = mri.replace('_flair.png', '_seg.npy')
            flair = mri
            seg = np.load(seg)
            #only healthy subjects
            if self.train:
                if np.unique(seg).size == 1:
                    self.lst.append([t1,flair,seg])
            else:
                if self.tumor:
                    if np.unique(seg).size != 1:
                        x = (seg > 0).astype(np.float16)
                        #return the number of nonzero elements
                        ood_proportion = np.count_nonzero(x)/(256**2)
                        if ood_proportion > 0.01:
                            self.lst.append([t1,flair,seg])
                        if len(self.lst) == 50:
                            break
                else:
                    if np.unique(seg).size == 1:
                        self.lst.append([t1,flair,seg])
                    if len(self.lst) == 50:
                        break

    def transform(self, img, img2, seg, size=(224,224)):
        if self.config['augmentations']: 
            if self.train:
                T = transforms.Compose([
                    transforms.CenterCrop(size),
                    transforms.RandomRotation(15),
                    transforms.RandomVerticalFlip()
                ])
            else:
                T = transforms.Compose([
                    transforms.CenterCrop(size)
                ])
            seed = torch.random.seed()  # Ensure random state is the same for both images
            torch.random.manual_seed(seed)
            img = T(img)
            torch.random.manual_seed(seed)
            img2 = T(img2)
            torch.random.manual_seed(seed)
            seg = T(seg)
            return img, img2, seg
        else:
            T = transforms.Compose([
                transforms.CenterCrop(size)
            ])
            img =  T(img)
            img2 = T(img2)
            seg = T(seg)
            #img = T.GaussianBlur(kernel_size=3, sigma=(0.7, 1.0))(img)
            #return T(img)
            return img, img2, seg
        
    def normalize(self, img, mode='t1'):
        if mode == 't1':
            img = (img - self.config['mean_t1'])/(self.config['std_t1'])
        else:
            img = (img - self.config['mean_flair'])/(self.config['std_flair'])

        if self.config['translate_zero']:
            mini = torch.abs(img.min())
            img = img + mini
        
        return img
    
    def np2tensor(self, x):
        x = torch.tensor(x)
        x = torch.unsqueeze(x,0)
        return x

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        t1, flair, seg = self.lst[idx][0], self.lst[idx][1], self.lst[idx][2]
        t1_sample = np.array(Image.open(t1)).astype(np.float32)#[:,:,0]
        flair_sample = np.array(Image.open(flair)).astype(np.float32)#[:,:,0]
        seg = torch.tensor(seg.astype(np.float32))

        t1_sample = self.np2tensor(t1_sample)
        flair_sample = self.np2tensor(flair_sample)
        seg = self.np2tensor(seg)

        t1_sample, flair_sample, seg = self.transform(t1_sample, flair_sample, seg)
        #t1_sample = self.transform(t1_sample)
        t1_sample = self.normalize(t1_sample, mode='t1')
        
        #flair_sample = self.np2tensor(flair_sample)
        #flair_sample = self.transform(flair_sample)
        flair_sample = self.normalize(flair_sample, mode='flair')

        #seg = self.np2tensor(seg)
        #seg = self.transform(seg)
  
        return t1_sample, flair_sample, seg 
      
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
            self.total = 28
            #only healthy subjects
            if self.train:
                for i in range(60,120,5):
                    slice = seg[:,:,i]
                    if np.unique(slice).size == 1:
                        self.lst.append([t1,flair,slice,i])
            else:
                if self.tumor:
                    low,high,skip = 60,120,5
                    for i in range(low, high, skip):
                        slice = seg[:,:,i]
                        if np.unique(slice).size != 1:
                            x = (slice > 0).astype(np.float16)
                            #return the number of nonzero elements
                            ood_proportion = np.count_nonzero(x)/(256**2)
         #                   if ood_proportion < 0.01:
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
                if len(self.lst) == self.total:
                    break

    def transform(self, img, size=(224,224)):
        
        img =  T.CenterCrop(size)(img)
        #img = T.GaussianBlur(kernel_size=3, sigma=(0.7, 1.0))(img)
        return img
        
    def normalize(self, img, mode='t1'):
        #img = (img - img.min()) / (img.max() - img.min())
        #img = ((img / 4096.0)*255).to(torch.uint8)
        #img = img / 255.0
        #img = 2*(img-0.5)
         
        if mode == 't1':
            img = (img - self.config['mean_t1'])/(self.config['std_t1'])
        else:
            img = (img - self.config['mean_flair'])/(self.config['std_flair'])
        
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

#IMAGE Segmentation for BRATS
class MedSegDataset(Dataset):
    def __init__(
        self,
        config,
        mri_files,
        train=True
    ):
        self.config = config
        self.mris = mri_files
        self.train = train
        self.lst = []
        for mri in self.mris:
            flair = glob.glob(mri + '/VSD.Brain.XX.O.MR_Flair/*.mha')[0]
            seg = glob.glob(mri + 'VSD.Brain_*more.XX*/*.mha')[0]

            seg, _ = load(seg)
            self.cnt = 0
            self.total = 28
            low,high,skip = 60,140,3
            for i in range(low, high, skip):
                slice = seg[:,:,i]
                if len(np.unique(slice)) > 1:
                    self.lst.append([flair,slice,i])
                #else:
                #    self.lst_healthy.append([flair,slice,i])

        #print(len(self.lst_tumor), len(self.lst_healthy))
        #self.lst = self.lst_tumor #+ self.lst_healthy
                

    def transform(self, img, size=(224,224)):
        transform = T.Compose([
            T.CenterCrop(size)])
            #T.RandomHorizontalFlip(),
            #T.RandomVerticalFlip()])
        return transform(img)
        
    def normalize(self, img):
        
        img = (img - self.config['mean_flair'])/(self.config['std_flair'])
        
        return img
    
    def np2tensor(self, x, unsqueeze=True):
        x = torch.tensor(x)
        if unsqueeze:
            x = torch.unsqueeze(x,0)
        return x

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        flair, seg, slice = self.lst[idx][0], self.lst[idx][1], self.lst[idx][2]
        flair_sample, _ = load(flair)
        flair_sample = flair_sample[:,:,slice].astype(np.float32)
        #make into 3-channel image
        #flair_sample = np.stack((flair_sample,flair_sample,flair_sample),axis=0)
        seg = torch.tensor((seg>0).astype(np.float32))
        
        flair_sample = self.np2tensor(flair_sample)
        flair_sample = self.transform(flair_sample)
        flair_sample = self.normalize(flair_sample)

        seg = self.np2tensor(seg)
        seg = self.transform(seg)
        
        return flair_sample, seg

#IMAGE Segmentation for BRATS
class MedSegDataset(Dataset):
    def __init__(
        self,
        config,
        mri_files,
        train=True
    ):
        self.config = config
        self.mris = mri_files
        self.train = train
        self.lst = []
        for mri in self.mris:
            flair = glob.glob(mri + '/VSD.Brain.XX.O.MR_Flair/*.mha')[0]
            seg = glob.glob(mri + 'VSD.Brain_*more.XX*/*.mha')[0]

            seg, _ = load(seg)
            self.cnt = 0
            self.total = 28
            low,high,skip = 60,140,3
            for i in range(low, high, skip):
                slice = seg[:,:,i]
                if len(np.unique(slice)) > 1:
                    self.lst.append([flair,slice,i])
                #else:
                #    self.lst_healthy.append([flair,slice,i])

        #print(len(self.lst_tumor), len(self.lst_healthy))
        #self.lst = self.lst_tumor #+ self.lst_healthy
                

    def transform(self, img, size=(224,224)):
        transform = T.Compose([
            T.CenterCrop(size)])
            #T.RandomHorizontalFlip(),
            #T.RandomVerticalFlip()])
        return transform(img)
        
    def normalize(self, img):
        
        img = (img - self.config['mean_flair'])/(self.config['std_flair'])
        
        return img
    
    def np2tensor(self, x, unsqueeze=True):
        x = torch.tensor(x)
        if unsqueeze:
            x = torch.unsqueeze(x,0)
        return x

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        flair, seg, slice = self.lst[idx][0], self.lst[idx][1], self.lst[idx][2]
        flair_sample, _ = load(flair)
        flair_sample = flair_sample[:,:,slice].astype(np.float32)
        #make into 3-channel image
        #flair_sample = np.stack((flair_sample,flair_sample,flair_sample),axis=0)
        seg = torch.tensor((seg>0).astype(np.float32))
        
        flair_sample = self.np2tensor(flair_sample)
        flair_sample = self.transform(flair_sample)
        flair_sample = self.normalize(flair_sample)

        seg = self.np2tensor(seg)
        seg = self.transform(seg)
        
        return flair_sample, seg


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
                            
    def transform(self, img, size=(28, 28)):
        #create transform for MNIST
        if self.train:
            if self.config['augmentations']:
                T = transforms.Compose([
                    transforms.RandomRotation(15),
                    transforms.RandomVerticalFlip()
                    #convert torch tensor to PIL image
                    #transforms.ToPILImage(),
                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                T = transforms.Compose([
                ])
        else:
            T = transforms.Compose([])
        return T(img)
    
    def np2tensor(self, x):
        x = torch.tensor(x).float().unsqueeze(0)
        return x
    
    def convert2rgb(self, x):
        x = np.stack((x,)*3, axis=0)
        return x
    def add_noise(self, x):
        x = x + torch.randn(x.size()) * self.std + self.mean
        return x
    
    def normalize(self, x):
        return 2*(x/255.0) #(x - x.min())/(x.max() - x.min())
        #return (x - self.config['mean_mnist'])/(self.config['std_mnist'])
    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        img, label = self.lst[idx][0], self.lst[idx][1]
        img = self.np2tensor(img)
        label = self.np2tensor(label)
        img = self.transform(img)
        #downsample img
        if len(img.shape) == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        elif len(img.shape) == 3:
            img = img.unsqueeze(0)
        #img_down = F.interpolate(img, size=(img.shape[-1]//2, img.shape[-1]//2), mode='bilinear', align_corners=False)
        img_down = img[:,::2,::2] #28x28 -> 14x14
        #upsample img
        img_down = F.interpolate(img_down, size=(img.shape[-1], img.shape[-1]), mode='bilinear', align_corners=False)
        img_down = self.normalize(img_down)
        img = self.normalize(img)

        if len(img.shape) == 4:
            img = img.squeeze(0)
        if len(img_down.shape) == 4:
            img_down = img_down.squeeze(0)
  
        return img, img_down, label

