import torch
from torch.nn import Conv2d
from torch import nn
from torchsummary import summary
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.down1 = self.UnetDoubleConv(3,64)
        self.down2 = self.UnetDoubleConv(64,128)
        self.down3 = self.UnetDoubleConv(128,256)
        self.down4 = self.UnetDoubleConv(256,512)
        self.down5 = self.UnetDoubleConv(512,1024)
        
        self.up1 = self.UnetDoubleConv(64,82)
        self.up2 = self.UnetDoubleConv(128,64)
        self.up3 = self.UnetDoubleConv(256,128)
        self.up4 = self.UnetDoubleConv(512,256)
        self.up5 = self.UnetDoubleConv(1024,512)
        
#         self.upConvT1 = nn.ConvTranspose2d(64,)
        # self.upConvT2 = nn.ConvTranspose2d(128,64,2,2)
        # self.upConvT3 = nn.ConvTranspose2d(256,128,2,2)
        # self.upConvT4 = nn.ConvTranspose2d(512,256,2,2)
        # self.upConvT5 = nn.ConvTranspose2d(1024,512,2,2)
        
        self.upConvT2 = self.UnetUpConv(128,64)
        self.upConvT3 = self.UnetUpConv(256,128)
        self.upConvT4 = self.UnetUpConv(512,256)
        self.upConvT5 = self.UnetUpConv(1024,512)

        self.softmax = torch.nn.Softmax(dim=1)

        self.chut = nn.Conv2d(82,82,1,padding='same')

        
    def UnetDoubleConv(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def UnetUpConv(self,in_channels,out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,2,2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
            
    
    def forward(self,x):
        # x = x.permute(0, 3, 1, 2)
        convOut1 = self.down1(x)
        maxPool1 = nn.MaxPool2d(2)(convOut1)
        convOut2 = self.down2(maxPool1)
        maxPool2 = nn.MaxPool2d(2)(convOut2)
        convOut3 = self.down3(maxPool2)
        # maxPool3 = nn.MaxPool2d(2)(convOut3)
        # convOut4 = self.down4(maxPool3)
        # maxPool4 = nn.MaxPool2d(2)(convOut4)
        # convOut5 = self.down5(maxPool4)
#         maxPool5 = nn.MaxPool2d(2)(convOut5)
        
        
        # upPool4 = self.upConvT4(convOut4)
        # upPool4Cat = torch.cat([upPool4,convOut3],dim=1)
        # incConvOut3 = self.up4(upPool4Cat)
        # upPool3 = self.upConvT3(incConvOut3)
        upPool3 = self.upConvT3(convOut3)
        upPool3Cat = torch.cat([upPool3,convOut2],dim=1)
        incConvOut2 = self.up3(upPool3Cat)
        upPool2 = self.upConvT2(incConvOut2)
        upPool2Cat = torch.cat([upPool2,convOut1],dim=1)
        incConvOut1 = self.up2(upPool2Cat)
        
        out1 = self.up1(incConvOut1)
        out = self.chut(out1)
        # out = self.softmax(out)
        # out = out.permute(0,3,1,2)
        
        # upPool3 = self.upConvT3(incConvOut3)
        # incConvOut2 = self.up2(upPool3)
        
        return(out)
        