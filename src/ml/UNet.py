from torch.nn import Module, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, Sigmoid, MaxPool2d, Sequential
from torch import cat
from torchsummary import summary

class UNet(Module):
    def __init__(self, device):
        super(UNet, self).__init__()
        self.device = device

        self.inconv = ConvBlock(3, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.outconv = Conv2d(64, 3, kernel_size=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)
        x = self.sigmoid(x)
        return x
    
    def summary(self):
        return summary(self.to(self.device), (3, 256, 256))
    
class ConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.main = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),

            Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.main(x)
    
class DownBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        
        self.main = Sequential(
            MaxPool2d(kernel_size=2),
            ConvBlock(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.main(x)

class UpBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        
        self.up = ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip_connection):
        x = self.up(x)
        x = cat((x, skip_connection), dim=1)
        x = self.conv(x)
        return x
    
