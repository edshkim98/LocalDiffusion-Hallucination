import torch
import torch.nn as nn
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, GroupNorm

group_num = 16

class BasicBlock(nn.Module):

    def __init__(self, input_dim, mid_dim, output_dim, stride=1, padding=1, groups=1, dilation=1, residual=True, reduction=16, se=False):
        super(BasicBlock, self).__init__()
        
        self.residual = residual
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.output_dim = output_dim
        self.se = se
        self.convblock = torch.nn.Sequential(  
 
            torch.nn.Conv2d(self.input_dim, self.mid_dim, kernel_size=3, stride=stride, padding=padding),
            torch.nn.GroupNorm(group_num, self.mid_dim),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(self.mid_dim, self.output_dim, kernel_size=3, padding=padding),
            torch.nn.GroupNorm(group_num, self.output_dim)
        )
        if self.residual:
            if self.input_dim != self.output_dim:
                self.identity = torch.nn.Sequential(
                    torch.nn.Conv2d(self.input_dim, self.output_dim, kernel_size=3, stride=stride, padding=padding),
                    torch.nn.GroupNorm(group_num, self.output_dim)
                )

        self.se = Squeeze_Excite_Block(self.output_dim, reduction) if self.se else None
        
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.convblock(x)
        if self.se:
            out = self.se(out)

        if self.residual:
            if self.input_dim != self.output_dim:
                identity = self.identity(x)
            out += identity

        out = self.relu(out)

        return out

class Squeeze_Excite_Block(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool3d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class Upsample(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel=3, stride=2, padding=1, output_padding=1, bias=True):
        super(Upsample, self).__init__()

        self.upsample = torch.nn.ConvTranspose3d(
            input_dim, output_dim, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        #self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        return self.upsample(x)

class Upsample_(torch.nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = torch.nn.Upsample(mode="trilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)
    
class ResUnet(torch.nn.Module):
    def __init__(self, data='mri'):
        super(ResUnet, self).__init__()
        
        if (data == 'mvtecSR'):
            self.in_channels = 3 #configs['model']['in_channel']
        else:
            self.in_channels = 1 #configs['model']['in_channel']
        self.filters = [32,32,64,128,256] #configs['model']['filters']
        self.stride = (1,1) #configs['model']['stride']
        #self.upsample = configs['model']['upsample']
        self.se = False #configs['model']['se']
        self.skip = True #configs['model']['residual']
        self.data = data

        #Downsampling
        self.residual_conv1 = torch.nn.Sequential(
            BasicBlock(self.in_channels , self.filters[0], self.filters[1], tuple(self.stride), 1, residual=self.skip, se=self.se))
        
        self.residual_conv2 = torch.nn.Sequential(
            BasicBlock(self.filters[1], self.filters[1], self.filters[2], tuple(self.stride), 1, residual=self.skip, se=self.se))
        
        self.residual_conv3 = torch.nn.Sequential(
            BasicBlock(self.filters[2], self.filters[2], self.filters[3], tuple(self.stride), 1, residual=self.skip, se=self.se))
        if (self.data == 'mri') or (data == 'mvtec'):
            self.mid_conv = torch.nn.Sequential(
                BasicBlock(self.filters[3], self.filters[3], self.filters[4], (1,1), 1, residual=self.skip, se=self.se))
        
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
                
        x2_conv = self.residual_conv1(x) #16
        x2 = self.maxpool(x2_conv)
        
        x3_conv = self.residual_conv2(x2) #8
        x3 = self.maxpool(x3_conv)

        x4_conv = self.residual_conv3(x3) #8
        if (self.data == 'mnist') or (self.data == 'mvtecSR'):
            return x4_conv
        x4 = self.maxpool(x4_conv)

        x4 = self.mid_conv(x4) #8

        return x4 #x4_conv
