import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp
import math

import os

import random
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from mnist_data import MNISTDataset
from model import UNet

import wandb
from PIL import Image
from tqdm import tqdm

# This is the amount of timesteps we can sample from
T = 4000

# A fixed hyperparameter for the cosine scheduler
S = 0.008

def linear_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)

def cosine_schedule(timesteps):

    max_beta = 0.999

    # The equation for f(t) to get alpha prod values
    f_t = lambda t: math.cos(((t / timesteps + S) / (S + 1)) * (math.pi / 2)) ** 2

    # Construct a list of beta values
    betas = [min(1 - f_t(i+1) / f_t(i), max_beta) for i in range(timesteps)]

    return torch.tensor(betas)


# This is our variance schedule
BETAS = linear_schedule(T)
ALPHAS = 1 - BETAS                                              # See paper
SQRT_RECIP_ALPHAS = 1.0 / torch.sqrt(ALPHAS)                    # Reciprocal of square root of alpha values                              
ALPHA_CUMPROD = torch.cumprod(ALPHAS, dim=0)                    # Cumulative product of the alpha values
SQRT_ALPHA_CUMPROD = torch.sqrt(ALPHA_CUMPROD)                  # Sqrt of the cumulative product of alpha values
SQRT_ONE_MINUS_ALPHA_CUMPROD = torch.sqrt(1 - ALPHA_CUMPROD)    # Sqrt of one minus alpha cumprod values


def reverse_transform(image):
    """ Takes in a tensor image and converts it to a uint8 numpy array"""

    image = (image + 1) / 2                 # Range [-1, 1] -> [0, 1)
    image *= 255                            # Range [0, 1) -> [0, 255)
    image = image.numpy().astype(np.uint8)  # Cast image to numpy and make it an unsigned integer type (no negatives)

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

def reverse_diff(model, device, image_size, image_channels, batch_size=1):
    

    with torch.no_grad():

        # Creates a random starting noise for T=1000
        img = torch.randn(batch_size, image_channels, image_size, image_size)

        # Construct a list of frames to visualize the model's progression
        frames = [img]

        for t in reversed(range(0, T)):

            # Create a random noise to add w/ variance
            z = torch.randn(batch_size, image_channels, image_size, image_size) if t > 1 else 0

            timesteps = torch.full((batch_size, 1), t)
            # Calculate the mean of the new img
            model_mean = SQRT_RECIP_ALPHAS[t] * (img - (BETAS[t] * model(img.to(device), timesteps.to(device)).cpu() / SQRT_ONE_MINUS_ALPHA_CUMPROD[t]))

            # Calculate the variance of the new image
            posterior_variance = math.sqrt(BETAS[t] * (1.0 - ALPHA_CUMPROD[t-1]) / (1.0 - ALPHA_CUMPROD[t]))

            # New image is mean + variance * random noise
            img = model_mean + posterior_variance * z
            
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

def construct_image_grid(model, device, image_size, image_channels, num_imgs):
    """Constructs a 3x3 grid of images using the diffusion model"""

    imgs = reverse_diff(model, device, image_size, image_channels, num_imgs)[-1]

    return make_grid(imgs, nrow=int(math.sqrt(num_imgs)))


def main():

    # Define Hyperparameters
    batch_size = 128
    epochs = 100
    lr = 2e-4
    device = 1
    channel_space = 64

    image_size = 28
    image_channels = 1
    dim_mults = (1, 2)

    n_log_images = 9        # How many images to log to wandb each epoch

    model_checkpoint = None

    # Define the dataset and dataloader
    train_set = MNISTDataset(T)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)

    # Define model, optimizer, and loss function
    model = UNet(img_start_channels = image_channels, channel_space=channel_space, dim_mults=dim_mults).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint))

    # Initialize wandb project
    wandb.init(project='diffusion_testing')

    # Keep track of training iterations
    count = 0
    for epoch in range(epochs):
        
        # Before each epoch, make sure model is in training mode
        model.train()

        # Initialize statistics to keep track of loss
        running_loss = 0

        # Iterate over batch of images and random timesteps
        for images, timesteps in tqdm(train_loader):

            # Define gaussian noise to be applied to this image
            noises = torch.randn_like(images)
            
            # Apply noise to each image at each timestep with each gaussian noise
            noised_images = apply_noise(images, timesteps, noises)

            # Cast inputs, targets, and timesteps to device
            noised_images, noises, timesteps = map(lambda x: x.to(device), [noised_images, noises, timesteps])

            # Predict the noise used to noise each image
            predicted_noise = model(noised_images, timesteps)

            # Apply loss to each prediction and update count
            loss = criterion(predicted_noise, noises)
            running_loss += loss.item()

            # Update model parameters
            optimizer.zero_grad()   # Zero out gradients
            loss.backward()         # Send loss backwards (compute gradients)
            optimizer.step()        # Update model weights
            count += 1
        
        # Set model to evaluation mode before doing validation steps
        model.eval()

        # Construct a grid of generated images to log and convert them to a wandb object
        image_array = construct_image_grid(model, device, image_size, image_channels, n_log_images)
        gen_images = wandb.Image(image_array, caption='Sampled images from diffusion model')

        # Grab random samples from the training set and convert them into wandb image for logging
        real_images = train_set[random.randint(0, len(train_set) - 1)][0]
        image = reverse_transform(real_images).squeeze()
        image = wandb.Image(Image.fromarray(image), caption="test")

        # Construct a gif of the diffusion process and turn it into a series of numpy images
        gif = reverse_transform(reverse_diff(model, device, image_size, image_channels))

        # Log all of the statistics to wandb
        wandb.log({'Loss': running_loss / len(train_loader),
                    'Training Iters': count,
                    'Generated Images': gen_images,
                    'Real Images': image,
                    'Gif': wandb.Video(gif, fps=60, format='gif')})

        # Save the model checkpoint somewhere
        torch.save(model.state_dict(), 'checkpoints/weights_1.pt')


if __name__ == "__main__":
    main()


