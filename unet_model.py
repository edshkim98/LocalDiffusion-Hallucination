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
    def __init__(self):
        super(ResUnet, self).__init__()
        
        self.in_channels = 1 #configs['model']['in_channel']
        self.filters = [32,32,64,128,256] #configs['model']['filters']
        self.stride = (1,1) #configs['model']['stride']
        #self.upsample = configs['model']['upsample']
        self.se = False #configs['model']['se']
        self.skip = True #configs['model']['residual']

        #Downsampling
        self.residual_conv1 = torch.nn.Sequential(
            BasicBlock(self.in_channels , self.filters[0], self.filters[1], tuple(self.stride), 1, residual=self.skip, se=self.se))
        
        self.residual_conv2 = torch.nn.Sequential(
            BasicBlock(self.filters[1], self.filters[1], self.filters[2], tuple(self.stride), 1, residual=self.skip, se=self.se))
        
        self.residual_conv3 = torch.nn.Sequential(
            BasicBlock(self.filters[2], self.filters[2], self.filters[3], tuple(self.stride), 1, residual=self.skip, se=self.se))

       # self.mid_conv = torch.nn.Sequential(
       #     BasicBlock(self.filters[3], self.filters[3], self.filters[4], (1,1), 1, residual=self.skip, se=self.se))
        
        self.maxpool = nn.MaxPool2d(2)
        '''
        #Upsampling
        if self.upsample == 'interpolation':
            self.upsample1 = Upsample_()
            self.upsample2 = Upsample_()
        elif self.upsample == 'transpose':
            self.upsample1 = Upsample(self.filters[4], self.filters[4])
            self.upsample2 = Upsample(self.filters[3], self.filters[3])
            self.upsample3 = Upsample(self.filters[2], self.filters[2])

        self.up_residual_conv1 = torch.nn.Sequential( 
            BasicBlock(self.filters[4] + self.filters[3], self.filters[3], self.filters[3], 1, 1, residual=self.skip, se=self.se),
        )
        self.up_residual_conv2 = torch.nn.Sequential( 
            BasicBlock(self.filters[3] + self.filters[2], self.filters[2], self.filters[2], 1, 1, residual=self.skip, se=self.se),
        )
        self.up_residual_conv3 = torch.nn.Sequential( 
            BasicBlock(self.filters[2] + self.filters[1], self.filters[1], self.filters[1], 1, 1, residual=self.skip, se=self.se),
        )

        self.output_layer = torch.nn.Sequential(
            torch.nn.Conv3d(self.filters[1], self.in_channels, kernel_size=1, padding=0),
        )
        '''       
    def forward(self, x):
                
        x2_conv = self.residual_conv1(x) #16
        x2 = self.maxpool(x2_conv)
        
        x3_conv = self.residual_conv2(x2) #8
        x3 = self.maxpool(x3_conv)

        x4_conv = self.residual_conv3(x3) #8
        #x4 = self.maxpool(x4_conv)

        #x4 = self.mid_conv(x4) #8
        '''
        x5 = self.upsample1(x4) #8
        x5 = torch.cat([x5, x4_conv], dim=1)
        x5 = self.up_residual_conv1(x5)
        
        x6 = self.upsample2(x5) #16
        x6 = torch.cat([x6, x3_conv], dim=1)
        x6 = self.up_residual_conv2(x6)

        x7 = self.upsample3(x6) #16
        x7 = torch.cat([x7, x2_conv], dim=1)
        x7 = self.up_residual_conv3(x7)
        
        out = self.output_layer(x7)
        '''
        return x4_conv
    
class Unet(torch.nn.Module):
    def __init__(self, configs):
        super(Unet, self).__init__()
        
        self.in_channels = configs['model']['in_channel']
        self.filters = configs['model']['filters']
        self.stride = [1,1,1]#configs['model']['stride']
        self.upsample = configs['model']['upsample']
        self.se = False #configs['model']['se']
        self.skip = False #configs['model']['residual']

        #Downsampling
        self.stem = torch.nn.Conv3d(self.in_channels, self.filters[0], kernel_size=3, padding=1)
        self.residual_conv1 = torch.nn.Sequential(
            BasicBlock(self.filters[0] , self.filters[0], self.filters[0], tuple(self.stride), 1, residual=self.skip, se=self.se))
        
        self.residual_conv2 = torch.nn.Sequential(
            BasicBlock(self.filters[0], self.filters[1], self.filters[1], tuple(self.stride), 1, residual=self.skip, se=self.se))
        
        self.mid_conv = torch.nn.Sequential(
            BasicBlock(self.filters[1], self.filters[2], self.filters[2], (1,1,1), 1, residual=self.skip, se=self.se))
        
        self.maxpool = nn.MaxPool3d(2)

        #Upsampling
        if self.upsample == 'interpolation':
            self.upsample1 = Upsample_()
            self.upsample2 = Upsample_()
        elif self.upsample == 'transpose':
            self.upsample1 = Upsample(self.filters[2], self.filters[2])
            self.upsample2 = Upsample(self.filters[1], self.filters[1])

        self.up_residual_conv1 = torch.nn.Sequential( 
            BasicBlock(self.filters[2] + self.filters[1], self.filters[1], self.filters[1], 1, 1, residual=self.skip, se=self.se),
        )
        self.up_residual_conv2 = torch.nn.Sequential( 
            BasicBlock(self.filters[1] + self.filters[0], self.filters[0], self.filters[0], 1, 1, residual=self.skip, se=self.se),
        )

        self.output_layer = torch.nn.Sequential(
            torch.nn.Conv3d(self.filters[0], self.in_channels, kernel_size=1, padding=0),
        )
                
    def forward(self, x):
        
        x1 = self.stem(x)
       
        x2_conv = self.residual_conv1(x1) #16
        x2 = self.maxpool(x2_conv)
        
        x3_conv = self.residual_conv2(x2) #8
        x3 = self.maxpool(x3_conv)

        x4 = self.mid_conv(x3) #8
        
        x5 = self.upsample1(x4) #8
        x5 = torch.cat([x5, x3_conv], dim=1)
        x5 = self.up_residual_conv1(x5)
        
        x6 = self.upsample2(x5) #16
        x6 = torch.cat([x6, x2_conv], dim=1)
        x6 = self.up_residual_conv2(x6)

        out = self.output_layer(x6)
        
        return out
    
#######################################################
# Implementation of 3D U-Net
#######################################################

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels == 1:
            self.mid_channels = self.out_channels//2
        else:
            self.mid_channels = self.in_channels
        self.block =  nn.Sequential(
        nn.Conv3d(self.in_channels, self.mid_channels, 3, padding=1),
        nn.BatchNorm3d(self.mid_channels),
        nn.ReLU(),
        nn.Conv3d(self.mid_channels, self.out_channels, 3, padding=1),
        nn.BatchNorm3d(self.out_channels),
        nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)   

class UNet3D(nn.Module):

    def __init__(self, in_channels, filters=[64,128,256,512]):
        super().__init__()
        
        self.filters = filters
        #self.init_conv = nn.Conv3d(in_channels, self.filters[0], kernel_size=3, padding=1)
        self.conv_down1 = double_conv(in_channels, self.filters[0])
        self.conv_down2 = double_conv(self.filters[0], self.filters[1])
        self.conv_down3 = double_conv(self.filters[1], self.filters[2])

        self.mid_conv = double_conv(self.filters[2], self.filters[3])

        self.maxpool = nn.MaxPool3d(2)

        self.upsample1 = Upsample(self.filters[3], self.filters[3])
        self.upsample2 = Upsample(self.filters[2], self.filters[2])
        self.upsample3 = Upsample(self.filters[1], self.filters[1])
        
        self.dconv_up3 = double_conv(self.filters[3] + self.filters[2], self.filters[2])
        self.doncv_up3_2 = double_conv(self.filters[2], self.filters[2])
        self.dconv_up2 = double_conv(self.filters[2] + self.filters[1], self.filters[1])
        self.doncv_up2_2 = double_conv(self.filters[1], self.filters[1])
        self.dconv_up1 = double_conv(self.filters[1] + self.filters[0], self.filters[0])
        self.dconv_up1_2 = double_conv(self.filters[0], self.filters[0])
 
        self.conv_last = nn.Conv3d(self.filters[0], in_channels, 1, padding=0)
        
        
    def forward(self, x):

        #x = self.init_conv(x)
        
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1) #16

        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2) #8
        
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3) #4

        x = self.mid_conv(x)
        
        x = self.upsample1(x) #8        
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.doncv_up3_2(x)

        x = self.upsample2(x) #16  
        x = torch.cat([x, conv2], dim=1)       
        x = self.dconv_up2(x)
        x = self.doncv_up2_2(x)

        x = self.upsample3(x) #32
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.dconv_up1_2(x)
        
        out = self.conv_last(x)
        
        return out
