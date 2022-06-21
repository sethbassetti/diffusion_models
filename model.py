import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.query_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Project keys, queries, values and reshape from B x N*C x H x W -> B x N x C x H*W
        queries = self.query_proj(x).reshape(batch_size, self.heads, -1, height * width)
        keys = self.key_proj(x).reshape(batch_size, self.heads, -1, height * width)
        values = self.value_proj(x).reshape(batch_size, self.heads, -1, height * width)

        # Scale down the queries
        queries = queries * self.scale

        # Multiply queries and keys together
        sim = torch.einsum("b h d i, b h d j -> b h i j", queries, keys)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # Multiply attention matrix and values together
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, values)
        out = out.reshape(batch_size, -1, height, width)
        return self.to_out(out)



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

    def __init__(self, in_channels, out_channels, time_emb_dim=128):
        super().__init__()

        # Main building block, increasing channel size
        self.block1 = ConvBlock(in_channels, out_channels)
        self.block2 = ConvBlock(out_channels, out_channels)

        self.time_mlp = nn.Sequential(nn.LeakyReLU(), nn.Linear(time_emb_dim, out_channels))


        # Residual connection block, reshapes in channels to out channels to add residual if channels are different
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb=None):
        """First sends x through a convolutional block. Then injects x with positional information. Then
        sends it through second convolutional block and adds a residual connection"""
        
        # Used to reshape the time embeddings after linear layer
        batch_size = x.shape[0]

        h = self.block1(x)
        # Project time embeddings into current channel dimension and reshape from t= b x c -> b x c x 1 x 1
        time_emb = self.time_mlp(time_emb).reshape(batch_size, -1, 1, 1) if time_emb is not None else 0
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):


    def __init__(self, img_start_channels=1, channel_space=64, time_emb_dim=64, T=1000, dim_mults=(1, 2, 2, 2)):
        super().__init__()


        self.channel_space = channel_space
        self.start_channels = img_start_channels
        self.dim_mults = dim_mults

        self.time_embedding = TimeEmbedding(time_emb_dim, T)

        # Builds the series of contrastive blocks leading to the bottleneck
        self.contrastives = self.build_contrastives()
    
        # Defines the channel dimension at the 
        bottleneck_depth = self.dim_mults[-1] * channel_space
        
        # Defines the two bottleneck layers and converts it into a sequential model
        self.bottleneck_1 = WideResBlock(bottleneck_depth, bottleneck_depth)
        self.bottleneck_2 = WideResBlock(bottleneck_depth, bottleneck_depth)

        # Builds the series of expansive blocks leading to the output
        self.expansives = self.build_expansives()

        # Output convolutional block that returns image to original channel dim
        self.output_conv = nn.Sequential(WideResBlock(self.channel_space, self.channel_space),
                                         nn.Conv2d(self.channel_space, self.start_channels, kernel_size=1))
        

    def build_contrastives(self):
        """ Builds a series of contrastive blocks to reduce spatial resolution while increasing channel resolution"""

        # Initialize a module list to contain all of the contrastive blocks
        blocks = nn.ModuleList()

        downsample_op = nn.MaxPool2d(2)

        # Images will start as 1 channel (grayscale) and turn into 64 feature maps
        in_channels = self.start_channels

        # Iterate through each level to construct a conv block and downsample operation
        for scale in self.dim_mults:
            
            # Follow the scaling specified in dim_mults list
            out_channels = self.channel_space * scale

            # Define the wide resnet blocks that comprise each layer
            res_block_1 = WideResBlock(in_channels, out_channels)
            res_block_2 = WideResBlock(out_channels, out_channels)
            res_block_3 = WideResBlock(out_channels, out_channels)

            attn = Attention(out_channels) if scale == 2 else nn.Identity()

            # Each block will consist of a convolutional block and a downsample operation
            block = nn.ModuleList([res_block_1, res_block_2, res_block_3, attn, downsample_op])
            blocks.append(block)

            # Set in channels to previous out channels
            in_channels = out_channels

        return blocks

    def build_expansives(self):
        """Builds a series of expansive blocks to upsample spatial resolution and downsample channel space"""

        # Initialize a module list to hold all of the expansive blocks
        blocks = nn.ModuleList()
        
        # Calculate how many channels will be coming out of the bottleneck based on the depth of the network
        in_channels = self.dim_mults[-1] * self.channel_space

        for scale in reversed(self.dim_mults):

            out_channels = scale * self.channel_space

            # 2d Transpose conv to double the spatial resolution of the image and halve feature dimension
            upsample_op = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

            res_block_1 = WideResBlock(out_channels * 2, out_channels)
            res_block_2 = WideResBlock(out_channels, out_channels)
            res_block_3 = WideResBlock(out_channels, out_channels)

            attn = Attention(out_channels) if scale == 2 else nn.Identity()

            # Create a block that holds the convolutional block and the upsampling operation
            blocks.append(nn.ModuleList([res_block_1, res_block_2, res_block_3, attn, upsample_op]))

            # Set the in channels to the previous out channels and halve the channel space
            in_channels = out_channels

        return blocks


    def forward(self, x, t):
        
        # This is the list of residuals to pass across the UNet
        h = [] 

        # Create a fixed time embedding from t
        t = self.time_embedding(t)
        
        # Iterate through each level of the contrastive portion of the model
        for block_1, block_2, block_3, attn, downsample_op in self.contrastives:

            # Send x through one contrastive level of model
            x = block_1(x, t)            # Send x through conv block
            x = block_2(x, t)
            x = block_3(x, t)
            x = attn(x)
            h.append(x)             # Append x to list of residuals
            x = downsample_op(x)    # Downsample x in spatial resolution

        # Send x through bottleneck at bottom of model
        x = self.bottleneck_1(x, t)
        x = self.bottleneck_2(x, t)

        # Iterate through the expansive, upsampling part of the model
        for block_1, block_2, block_3, attn, upsample_op in self.expansives:
            # Send x through one expansive level of model
            x = upsample_op(x)                      # Upsample x to double spatial resolution
            x = torch.cat((h.pop(), x), dim=1)      # Concatenate residual connection to start of x
            x = block_1(x, t)                       # Send x through conv block
            x = block_2(x, t)
            x = block_3(x, t)
            x = attn(x)
            
            

        # Send x through output block that squashes channel space and return output
        out = self.output_conv(x)
        return out


if __name__ == "__main__":
    img = torch.randn(4, 3, 32, 32)
    timesteps = torch.randint(0, 1000, (4,))

    model = UNet(img_start_channels=3, channel_space=128, time_emb_dim=128)
    out = model(img, timesteps)
    print(out.shape)