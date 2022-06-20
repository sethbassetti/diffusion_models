import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

from mnist_data import MNISTDataset
from model import UNet

import wandb
from tqdm import tqdm

# This is the amount of timesteps we can sample from
T = 1000

# This is our variance schedule
BETAS = torch.linspace(0.0001, 0.02, T)
ALPHAS = 1 - BETAS                                              # See paper
SQRT_RECIP_ALPHAS = 1.0 / torch.sqrt(ALPHAS)                    # Reciprocal of square root of alpha values                              
ALPHA_CUMPROD = torch.cumprod(ALPHAS, dim=0)                    # Cumulative product of the alpha values
SQRT_ALPHA_CUMPROD = torch.sqrt(ALPHA_CUMPROD)                  # Sqrt of the cumulative product of alpha values
SQRT_ONE_MINUS_ALPHA_CUMPROD = torch.sqrt(1 - ALPHA_CUMPROD)    # Sqrt of one minus alpha cumprod values


def reverse_transform(image):
    """ Takes in a tensor image and converts it to a PIL image"""

    image = (image + 1) / 2

    image *= 255


    image = image.numpy().astype(np.uint8)

    return image

def forward_diff(image, t, noise=None):
    """ Given an image and a timestep (and optional noise), subjects the image to t levels of noising"""

    t = int(t)

    # If noise is not given, then just initialize it to gaussian noise in the shape of the image
    if noise is None:
        noise = torch.randn_like(image)

    # The new (noised) image will have a mean centered around the original image times the alpha value
    mean = SQRT_ALPHA_CUMPROD[t] * image

    # The variance will be controlled by the multiplication of the beta schedules
    variance = SQRT_ONE_MINUS_ALPHA_CUMPROD[t]

    # Push the gaussian distribution to center on our image with the appropriate variance
    noised_image = noise * variance + mean

    return noised_image


def apply_noise(images, timesteps, noises):
    """Takes in a batch of images, timesteps, and gaussian noises and uses the forward diffusion pass 
    to apply noise to them"""

    # Using each image, timestep, and noise in the batch run them through the forward diffusion pass to sample them
    noised_images = [forward_diff(image, t, noise=noise) for image, t, noise in zip(images, timesteps, noises)]
    
    # Return a tensor of images, instead of a list
    return torch.stack(noised_images)

def reverse_diff(model, device, image_size, image_channels):
    
    with torch.no_grad():
        # Creates a random starting noise for T=1000
        img = torch.randn(1, image_channels, image_size, image_size)

        # Construct a list of frames to visualize the model's progression
        frames = [img]

        for t in reversed(range(0, T)):

            # Create a random noise to add w/ variance
            z = torch.randn(1, image_channels, image_size, image_size) if t > 1 else 0

            # Calculate the mean of the new img
            model_mean = SQRT_RECIP_ALPHAS[t] * (img - (BETAS[t] * model(img.to(device), torch.tensor([t])).cpu() / SQRT_ONE_MINUS_ALPHA_CUMPROD[t]))

            # Calculate the variance of the new image
            variance = torch.sqrt(BETAS[t])

            # New image is mean + variance * random noise
            img = model_mean + variance * z
            
            frames.append(img)

    # Return the list of each timestep and gets rid of one of the extra dimensions
    frames = torch.stack(frames).squeeze(1)
    return frames

def linear_beta_schedule(t):
    """Linear schedule of beta coefficients for variables for forward diffusion process"""
    return torch.linspace(0.0001, 0.02, t)

def imshow(image):
    """Helper function for displaying images"""
    plt.imshow(image, cmap='gray')
    plt.show()

def construct_image_grid(model, device, image_size, image_channels):
    """Constructs a 3x3 grid of images using the diffusion model"""

    imgs = []

    # Make a list of 9 images to make a 3x3 grid
    for i in range(9):

        # Take the last frame in the reverse diffusion process and append it to the list
        img = reverse_diff(model, device, image_size, image_channels)[-1]
        imgs.append(img)

    # Convert the list into a tensor and use the make_grid() function to make a grid of images
    imgs = torch.stack(imgs)
    return make_grid(imgs, nrow=3)
    
def main():

    # Define Hyperparameters
    batch_size = 128
    epochs = 40
    lr = 2e-4
    device = 0
    depth = 2

    image_size = 28
    image_channels = 1

    # Define the dataset and dataloader
    train_set = MNISTDataset(T)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    # Define model, optimizer, and loss function
    model = UNet(depth, T=T, img_start_channels = image_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Initialize wandb project
    wandb.init(project='diffusion_testing')

    for epoch in range(epochs):

        # Initialize statistics to keep track of loss
        running_loss = 0

        # Iterate over batch of images and random timesteps
        for images, timesteps in tqdm(train_loader):

            # Define gaussian noise to be applied to this image
            noises = torch.randn_like(images)
            
            # Apply noise to each image at each timestep with each gaussian noise
            noised_images = apply_noise(images, timesteps, noises)

            # Cast inputs and targets to device
            noised_images, noises = noised_images.to(device), noises.to(device)
    
            # Predict the noise used to noise each image
            predicted_noise = model(noised_images, timesteps)

            # Apply loss to each prediction and update count
            loss = criterion(predicted_noise, noises)
            running_loss += loss.item()

            # Update model parameters
            optimizer.zero_grad()   # Zero out gradients
            loss.backward()         # Send loss backwards (compute gradients)
            optimizer.step()        # Update model weights
                
        # Construct a grid of generated images to log and convert them to a wandb object
        image_array = construct_image_grid(model, device, image_size, image_channels)
        images = wandb.Image(image_array, caption='Sampled images from diffusion model')

        # Construct a gif of the diffusion process and turn it into a series of numpy images
        gif = reverse_transform(reverse_diff(model, device, image_size, image_channels))

        # Log all of the statistics to wandb
        wandb.log({'Loss': running_loss / len(train_loader),
                    'Images': images,
                    'Gif': wandb.Video(gif, fps=60, format='gif')})


if __name__ == "__main__":
    main()


