import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if 'mvtecGray' in data:
            self.in_channels = 1 
        elif 'mvtec' in data:
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
        if (self.data == 'mri') or (data == 'mvtec') or (data =='mvtecGray'):
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

########################################
#segmentation

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
