import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn as nn

T = 1000

class MNISTDataset(Dataset):
    """Dataset class that holds MNIST images for diffusion model training"""

    def __init__(self):
        super().__init__()

        # Loads the data into memory
        self.images = self.load_data()

    def load_data(self):
        """Performs the loading of the data and all normalization/standardization"""
        
        # Loads the MNIST dataset, downloading if it is not already downloaded
        dataset = datasets.MNIST('./data/', train=True, download=True)

        # Converts images from [0, 255) integer scale to [0,1) float scale
        images = dataset.data / 255

        # Z-score normalization: (x-mean) / std now in [-1, 1] scale
        images = (images - images.mean())/images.std()

        # Adds a channel dimension to the images
        images = images.unsqueeze(1)

        return images

    def __getitem__(self, idx):

        # Grabs an image from the data
        image = self.images[idx]

        # Grabs a random timestep from 1 to the total number of timesteps 
        timestep = torch.randint(1, T, (1,))

        return image, timestep

    def __len__(self):
        return len(self.images)



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



def forward_diffusion_pass(image, t, noise=None):
    """ Noises an image using a specified variance schedule and a timestep. The equation for the noised image
   at a timestep is N( sqrt(prod(1-betas) * image), (1-prod(1-betas) * I)"""


    if noise == None:
        # Construct gaussian noise in the shape of our image
        noise = torch.randn_like(image)

    # A list of beta values representing the coefficient for the variance at each time step
    betas = linear_beta_schedule(t)

    # Construct the list of alphas and multiply all of them together to form a single variance coefficient
    alphas = [(1 - beta) for beta in betas]
    alpha_prod = math.prod(alphas)

    # The new (noised) image will have a mean centered around the original image times the alpha value
    normal_mean = math.sqrt(alpha_prod) * image

    # The variance will be controlled by the multiplication of the beta schedules
    normal_variance = (1 - alpha_prod)

    # Push the gaussian distribution to center on our image with the appropriate variance
    noised_image = noise * normal_variance + normal_mean
    return noised_image

def apply_noise(images, timesteps, noises):
    """Takes in a batch of images, timesteps, and gaussian noises and uses the forward diffusion pass 
    to apply noise to them"""

    # Using each image, timestep, and noise in the batch run them through the forward diffusion pass to sample them
    noised_images = [forward_diffusion_pass(image, t, noise=noise) for image, t, noise in zip(images, timesteps, noises)]
    
    # Return a tensor of images, instead of a list
    return torch.stack(noised_images)

def linear_beta_schedule(t):
    """Linear schedule of beta coefficients for variables for forward diffusion process"""
    return np.linspace(0.0001, 0.02, t)

def imshow(image):
    """Helper function for displaying images"""
    plt.imshow(image, cmap='gray')
    plt.show()



def main():

    # Define Hyperparameters
    batch_size = 16
    epochs = 5
    lr = 0.001


    # Define the dataset and dataloader
    train_set = MNISTDataset()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    # Define model, optimizer, and loss function
    model = UNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()


    for epoch in range(epochs):

        # Iterate over batch of images and random timesteps
        for images, timesteps in train_loader:

            # Define gaussian noise to be applied to this image
            noises = torch.randn_like(images)
            
            # Apply noise to each image at each timestep with each gaussian noise
            noised_images = apply_noise(images, timesteps, noises)

            # Predict the noise used to noise each image
            predicted_noise = model(noised_images)

            # Apply loss to each prediction
            loss = criterion(predicted_noise, noises)

            # Update model parameters
            optimizer.zero_grad()   # Zero out gradients
            loss.backward()         # Send loss backwards (compute gradients)
            optimizer.step()        # Update model weights


            print(loss) 

if __name__ == "__main__":
    main()


