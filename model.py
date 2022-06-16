import torch.nn as nn
import torch

class UNet(nn.Module):

    class ConvBlock(nn.Module):
        """ The main building block of the UNet Architecture. Consists of two
        convolutional layers with LeakyReLU activation function"""

        def __init__(self, in_channels, out_channels):
            super().__init__()

            # Main building block, increasing channel size
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
                nn.LeakyReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
                nn.LeakyReLU()
            )
        
        def forward(self, x):
            out = self.block(x)
            return out

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.downsample = nn.MaxPool2d(2)
        
        # Depth 1 contrastive block
        self.contractive_1 = UNet.ConvBlock(1, 64)
        
        # Depth 2 contrastive block
        self.contractive_2 = UNet.ConvBlock(64, 128)

        # Bottleneck block between contrastive and expansive blocks
        self.bottleneck = UNet.ConvBlock(128, 256)

        # Upsample layer between bottleneck and first expansive layer
        self.upsample_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        # Depth 2 expansive block
        self.expansive_2 = UNet.ConvBlock(256, 128)
        
        # Upsample layer between depth 2 expansive and depth 1 expansive
        self.upsample_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Depth 1 expansive block
        self.expansive_1 = UNet.ConvBlock(128, 64)

        # Output convolutional block
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)
        

    def forward(self, x):

        res_1 = self.contractive_1(x)

        x = self.downsample(res_1)
        res_2 = self.contractive_2(x)
        x = self.downsample(res_2)
        x = self.bottleneck(x)

        x = self.upsample_1(x)
        x = torch.cat((res_2, x), dim=1)

        x = self.expansive_2(x)
        x = self.upsample_2(x)
        x = torch.cat((res_1, x), dim=1)
        x = self.expansive_1(x)
        out = self.output_conv(x)
        return out
