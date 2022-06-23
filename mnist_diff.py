import torch
import torch.nn as nn
import torch.nn.functional as F
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
T = 300

# A fixed hyperparameter for the cosine scheduler
S = 0.008

def linear_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)

def cosine_schedule(timesteps):

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    alphas_cumprod = torch.cos(((x / timesteps) + S) / (1 + S) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0.0001, 0.9999)



# This is our variance schedule
BETAS = linear_schedule(T)
ALPHAS = 1 - BETAS                                              # See paper
SQRT_RECIP_ALPHAS = 1.0 / torch.sqrt(ALPHAS)                    # Reciprocal of square root of alpha values                              
ALPHA_CUMPROD = torch.cumprod(ALPHAS, dim=0)                    # Cumulative product of the alpha values
ALPHA_CUMPROD_PREV = F.pad(ALPHA_CUMPROD[:-1],(1, 0),value=1.0) # Pads the last element with a 1
SQRT_ALPHA_CUMPROD = torch.sqrt(ALPHA_CUMPROD)                  # Sqrt of the cumulative product of alpha values
SQRT_ONE_MINUS_ALPHA_CUMPROD = torch.sqrt(1 - ALPHA_CUMPROD)    # Sqrt of one minus alpha cumprod values

POSTERIOR_VARIANCE = BETAS * (1.0 - ALPHA_CUMPROD_PREV) / (1.0 - ALPHA_CUMPROD)

def reverse_transform(image):
    """ Takes in a tensor image and converts it to a uint8 numpy array"""

    image = (image + 1) / 2                 # Range [-1, 1] -> [0, 1)
    image *= 255                            # Range [0, 1) -> [0, 255)
    image = image.numpy().astype(np.uint8)  # Cast image to numpy and make it an unsigned integer type (no negatives)

    return image

def forward_diff(image, t, noise=None):
    """ Given an image and a timestep (and optional noise), subjects the image to t levels of noising"""

    # If noise is not given, then just initialize it to gaussian noise in the shape of the image
    if noise is None:
        noise = torch.randn_like(image)

    sqrt_alpha_cumprod = extract(SQRT_ALPHA_CUMPROD, t, image.shape)
    sqrt_one_minus_alpha_cumprod = extract(SQRT_ONE_MINUS_ALPHA_CUMPROD, t, image.shape)

    # The new (noised) image will have a mean centered around the original image times the alpha value
    mean = sqrt_alpha_cumprod * image

    # The variance will be controlled by the multiplication of the beta schedules
    variance = sqrt_one_minus_alpha_cumprod

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

def p_sample(model, x, t, t_index):

    # Extract each constant and reshape to shape of image
    betas = extract(BETAS, t, x.shape)
    sqrt_recip_alphas = extract(SQRT_RECIP_ALPHAS, t, x.shape)
    sqrt_one_minus_alpha_cumprod = extract(SQRT_ONE_MINUS_ALPHA_CUMPROD, t, x.shape)

    # Calculate the mean of the new img
    model_mean = sqrt_recip_alphas * (x - (betas * model(x, t) / sqrt_one_minus_alpha_cumprod))

    # If at the last timestep, just return mean without any noise
    if t_index == 0:
        return model_mean
    else:
        # Otherwise calculate posterior variance, random noise, and return that added to the predicted mean
        posterior_variance = extract(POSTERIOR_VARIANCE, t, x.shape)
        noise = torch.randn_like(x)

        return model_mean + torch.sqrt(posterior_variance) * noise


@torch.no_grad()
def reverse_diff(model, shape):
    """Constructs a sequence of frames of the denoising process"""
    device = next(model.parameters()).device

    b = shape[0]

    # Start imgs out with random noise
    img = torch.randn(shape, device=device)

    imgs = []
    
    # Loops through each timestep to get to the generated image
    for i in tqdm(reversed(range(0, T)), desc='sampling loop time step', total=T):

        # Samples from the specific timestep
        img = p_sample(model, img, torch.full((b,), i, device=device), i)
        imgs.append(img.cpu())

    return imgs



def imshow(image):
    """Helper function for displaying images"""
    plt.imshow(image, cmap='gray')
    plt.show()

def extract(a, t, x_shape):
    """ Helper function to extract indices from a tensor and reshape them"""

    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

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
    train_set = MNISTDataset()
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)

    # Define model, optimizer, and loss function
    model = UNet(img_start_channels = image_channels, channel_space=channel_space, dim_mults=dim_mults).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint))

    # Initialize wandb project
    wandb.init(project='diffusion_testing', settings=wandb.Settings(start_method='fork'))

    # Keep track of training iterations
    count = 0
    for epoch in range(epochs):
        
        # Before each epoch, make sure model is in training mode
        model.train()

        # Initialize statistics to keep track of loss
        running_loss = 0

        # Iterate over batch of images and random timesteps
        for images, _ in tqdm(train_loader, desc=f'Epoch {epoch}'):
            
            # Cast image to device
            images = images.to(device)

            # Define gaussian noise to be applied to this image
            noises = torch.randn_like(images)

            # Define a series of random timesteps to sample noise from
            timesteps = torch.randint(0, T, (images.shape[0], ), device=device, dtype=torch.long)
            
            # Apply noise to each image at each timestep with each gaussian noise
            noised_images = forward_diff(images, timesteps, noises)

            # Predict epsilon, the noise used to noise each image
            predicted_noise = model(noised_images, timesteps)

            # Apply loss to each prediction and update count
            loss = criterion(predicted_noise, noises)
            running_loss += loss.item()

            # Update model parameters
            optimizer.zero_grad()   # Zero out gradients
            loss.backward()         # Send loss backwards (compute gradients)
            optimizer.step()        # Update model weights
            count += 1              # Keep track of num of updates
        
        # Set model to evaluation mode before doing validation steps
        model.eval()

       

        # Grab random samples from the training set and convert them into wandb image for logging
        real_images = torch.stack([train_set[random.randint(0, len(train_set)-1)][0] for _ in range(n_log_images)])
        #for i in range(n_log_images):
            #real_images.append(train_set[random.randint(0, len(train_set) - 1)][0])
        #real_images = torch.stack(real_images)

        real_img_grid = make_grid(real_images, nrow=3, normalize=True)

        # Generate a batch of images
        gen_imgs = reverse_diff(model, (n_log_images, image_channels, image_size, image_size))

        # Make the last (t=0) slice of images into a grid
        gen_img_grid = make_grid(gen_imgs[-1], nrow=3, normalize=True)

        # Take all of the frames from the reverse diffusion process for an image and convert it into numpy array
        gif = reverse_transform(torch.stack(gen_imgs)[:,0])

        # Log all of the statistics to wandb
        wandb.log({'Loss': running_loss / len(train_loader),
                    'Training Iters': count,
                    'Generated Images': wandb.Image(gen_img_grid, caption="Generated Images"),
                    'Real Images': wandb.Image(real_img_grid, caption='Real Images'),
                    'Gif': wandb.Video(gif, fps=60, format='gif')})

        # Save the model checkpoint somewhere
        torch.save(model.state_dict(), 'checkpoints/weights_1.pt')


if __name__ == "__main__":
    main()


