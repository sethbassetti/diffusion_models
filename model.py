import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    """ Self Attention Layer between images/feature maps"""

    def __init__(self, in_channels):
        super().__init__()

        self.activation = nn.LeakyReLU()

        # Query and key convolutions transform image from B x C x H x W to B x C//8 x H x W
        self.query_conv = nn.Conv2d(in_channels, in_channels// 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels// 8, kernel_size=1)

        # Value conv retains channel dimension
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # A learned parameter denoting how much of the attention matrix to use vs. original feature maps
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Projects x to key, query, and value dimensions. Computes self attention and returns a mix of
        the attended x and the original x"""

        # Keep track of original dimensions
        batch_size, channels, height, width = x.size()

        # Project x to query dimension and reshape from B x C x H x W -> B x H*W x C
        query_proj = self.query_conv(x).reshape(batch_size, -1, height * width).permute(0, 2, 1)

        # Project x to key and value dimensions and reshape to B x C x H*W
        key_proj = self.key_conv(x).reshape(batch_size, -1, height * width)
        value_proj = self.value_conv(x).reshape(batch_size, -1, height * width)

        # Apply matrix multiplication between queries and keys and softmax it to derive attention weights
        attn_scores = torch.bmm(query_proj, key_proj)
        attn_weights = self.softmax(attn_scores)

        # Multiply attention weights and original values to get weighted values
        out = torch.bmm(value_proj, attn_weights.permute(0, 2, 1))
        out = out.reshape(batch_size, channels, height, width)

        # How much of the attended values to use vs. the original values is learned
        out = self.gamma * out + x
        return out

class TimeEmbedding(nn.Module):
    """A module that converts a timestep into an embedding vector"""

    def __init__(self, dim, T):

        super().__init__()
        # Dimensionality of the time embedding vector
        self.dim = dim

        # The maximum timestep this can hold
        self.max_timestep = T

        # Create matrices holding 0...T and 0...dim 
        timesteps = torch.arange(0, T).unsqueeze(1)
        dim_indices = torch.arange(0, dim).unsqueeze(0)

        # This is the term that position will be divided by
        div_term = 10000 ** (2 * dim_indices / dim)

        # Multiplies all of the positions (timesteps) to obtain a T x dim matrix
        pe_matrix = timesteps / div_term

        # Convert even indices to sin and odd indices to cos
        pe_matrix[:, 0::2] = torch.sin(pe_matrix[:, 0::2])
        pe_matrix[:, 1::2] = torch.cos(pe_matrix[:, 1::2])
        
        # Register this as a (non-learnable) parameter to this module
        self.register_buffer('pe_matrix', pe_matrix)

    def forward(self, t):
        """Looks up the timestep embedding for a specific timestep, t""" 

        return self.pe_matrix[t]


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
    """ The main building block of the UNet Architecture. Uses residual connections and sends x through
    two convolutional blocks, along with inserting the time embedding into it"""

    def __init__(self, in_channels, out_channels, time_emb_dim=64):
        super().__init__()

        # Main building block, increasing channel size
        self.block1 = ConvBlock(in_channels, out_channels)
        self.block2 = ConvBlock(out_channels, out_channels)

        self.time_mlp = nn.Sequential(nn.LeakyReLU(), nn.Linear(time_emb_dim, out_channels))


        # Residual connection block, reshapes in channels to out channels to add residual if channels are different
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb):
        """First sends x through a convolutional block. Then injects x with positional information. Then
        sends it through second convolutional block and adds a residual connection"""
        
        # Used to reshape the time embeddings after linear layer
        batch_size = x.shape[0]

        h = self.block1(x)

        # Project time embeddings into current channel dimension and reshape from t= b x c -> b x c x 1 x 1
        h = h + self.time_mlp(time_emb).reshape(batch_size, -1, 1, 1)
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):


    def __init__(self, depth, img_start_channels=1, channel_space=64, time_emb_dim=64, T=1000):
        super().__init__()

        self.channel_space = channel_space

        self.time_embedding = TimeEmbedding(time_emb_dim, T)

        # Builds the series of contrastive blocks leading to the bottleneck
        self.contrastives = self.build_contrastives(depth)
    
        # Defines the channel dimension at the 
        bottleneck_depth = self.channel_space * 2 ** (depth - 1)
        
        # Defines the two bottleneck layers and converts it into a sequential model
        self.bottleneck_1 = WideResBlock(bottleneck_depth, bottleneck_depth * 2)
        self.bottleneck_2 = WideResBlock(bottleneck_depth * 2, bottleneck_depth * 2)

        # Builds the series of expansive blocks leading to the output
        self.expansives = self.build_expansives(depth)

        # Output convolutional block that returns image to original channel dim
        self.output_conv = nn.Conv2d(self.channel_space, img_start_channels, kernel_size=1)
        

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
            
            # Define the wide resnet blocks that comprise each layer
            res_block_1 = WideResBlock(in_channels, out_channels)
            res_block_2 = WideResBlock(out_channels, out_channels)

            # Each block will consist of a convolutional block and a downsample operation
            block = nn.ModuleList([res_block_1, res_block_2, downsample_op])
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
            upsample_op = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

            conv_block_1 = WideResBlock(in_channels, out_channels)

            conv_block_2 = WideResBlock(out_channels, out_channels)

            # Create a block that holds the convolutional block and the upsampling operation
            blocks.append(nn.ModuleList([conv_block_1, conv_block_2, upsample_op]))

            # Set the in channels to the previous out channels and halve the channel space
            in_channels = out_channels
            out_channels //= 2

        return blocks


    def forward(self, x, t):
        
        # This is the list of residuals to pass across the UNet
        h = [] 

        # Create a fixed time embedding from t
        t = self.time_embedding(t)
        
        # Iterate through each level of the contrastive portion of the model
        for block_1, block_2, downsample_op in self.contrastives:

            # Send x through one contrastive level of model
            x = block_1(x, t)            # Send x through conv block
            x = block_2(x, t)
            if x.shape[-1] == 16:
                channels = x.shape[1]
                attn = SelfAttention(channels)
                x = attn(x)
            h.append(x)             # Append x to list of residuals
            x = downsample_op(x)    # Downsample x in spatial resolution

        # Send x through bottleneck at bottom of model
        x = self.bottleneck_1(x, t)
        x = self.bottleneck_2(x, t)

        # Iterate through the expansive, upsampling part of the model
        for block_1, block_2, upsample_op in self.expansives:

            # Send x through one expansive level of model
            x = upsample_op(x)                      # Upsample x to double spatial resolution
            x = torch.cat((h.pop(), x), dim=1)      # Concatenate residual connection to start of x
            x = block_1(x, t)                       # Send x through conv block
            x = block_2(x, t)

            # If we are at the 16 x 16 spatial resolution, apply self attention
            if x.shape[-1] == 16:
                channels = x.shape[1]
                attn = SelfAttention(channels)
                x = attn(x)
            
            

        # Send x through output block that squashes channel space and return output
        out = self.output_conv(x)
        return out


if __name__ == "__main__":
    img = torch.randn(4, 1, 28, 28)
    timesteps = torch.randint(0, 1000, (4,))

    model = UNet(2)
    out = model(img, timesteps)
    print(out.shape)