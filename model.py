import torch.nn as nn
import torch
import math

class UNet(nn.Module):

    class ConvBlock(nn.Module):
            """Block used within the resnet, a simple convolutional layer with groupnorm and leaky relu"""

            def __init__(self, in_channels, out_channels):
                super().__init__()

                # Defines the main conv layer building block of the resnet
                self.block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
                    nn.GroupNorm(8, out_channels),
                    nn.LeakyReLU()
                )

            def forward(self, x):
                out = self.block(x)
                return out


    class WideResBlock(nn.Module):
        """ The main building block of the UNet Architecture. Consists of two
        convolutional layers with LeakyReLU activation function"""

        def __init__(self, in_channels, out_channels):
            super().__init__()

            # Main building block, increasing channel size
            self.block1 = UNet.ConvBlock(in_channels, out_channels)
            self.block2 = UNet.ConvBlock(out_channels, out_channels)

            # Residual connection block, reshapes in channels to out channels to add residual if channels are different
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        def forward(self, x):
            h = self.block1(x)
            h = self.block2(h)
            return h + self.res_conv(x)

    def __init__(self, depth):
        super().__init__()

        
        # Builds the series of contrastive blocks leading to the bottleneck
        self.contrastives = self.build_contrastives(depth)
    
        # Bottleneck block between contrastive and expansive blocks
        bottleneck_depth = 64 * 2 ** (depth - 1)
        self.bottleneck = UNet.WideResBlock(bottleneck_depth, bottleneck_depth * 2)

        # Builds the series of expansive blocks leading to the output
        self.expansives = self.build_expansives(depth)

        # Output convolutional block
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)

    def build_contrastives(self, depth):
        """ Builds a series of contrastive blocks to reduce spatial resolution while increasing channel resolution"""

        # Initialize a module list to contain all of the contrastive blocks
        blocks = nn.ModuleList()

        downsample_op = nn.MaxPool2d(2)

        # Images will start as 1 channel (grayscale) and turn into 64 feature maps
        in_channels = 1
        out_channels = 64

        # Iterate through each level to construct a conv block and downsample operation
        for _ in range(depth):

            # Each block will consist of a convolutional block and a downsample operation
            block = nn.ModuleList([UNet.WideResBlock(in_channels, out_channels), downsample_op])
            blocks.append(block)

            # Set in channels to previous out channels and double out channel space
            in_channels = out_channels
            out_channels *= 2

        return blocks

    def build_expansives(self, depth):
        """Builds a series of expansive blocks to upsample spatial resolution and downsample channel space"""

        # Initialize a module list to hold all of the expansive blocks
        blocks = nn.ModuleList()
        
        # Calculate how many channels will be coming out of the bottleneck based on the depth of the network
        in_channels = 64 * 2 ** (depth)
        out_channels = 64 * 2 ** (depth-1)

        for _ in reversed(range(depth)):

            # 2d Transpose conv to double the spatial resolution of the image and halve feature dimension
            upsample_op = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

            conv_block = UNet.WideResBlock(in_channels, out_channels)

            # Create a block that holds the convolutional block and the upsampling operation
            blocks.append(nn.ModuleList([conv_block, upsample_op]))

            # Set the in channels to the previous out channels and halve the channel space
            in_channels = out_channels
            out_channels //= 2

        return blocks


    def forward(self, x):
        
        # This is the list of residuals to pass across the UNet
        h = []

        # Iterate through each level of the contrastive portion of the model
        for block, downsample_op in self.contrastives:

            # Send x through one contrastive level of model
            x = block(x)            # Send x through conv block
            h.append(x)             # Append x to list of residuals
            x = downsample_op(x)    # Downsample x in spatial resolution

        # Send x through bottleneck at bottom of model
        x = self.bottleneck(x)

        # Iterate through the expansive, upsampling part of the model
        for block, upsample_op in self.expansives:

            # Send x through one expansive level of model
            x = upsample_op(x)                  # Upsample x to double spatial resolution
            x = torch.cat((h.pop(), x), dim=1)  # Concatenate residual connection to start of x
            x = block(x)                        # Send x through conv block

        # Send x through output block that squashes channel space and return output
        out = self.output_conv(x)
        return out
