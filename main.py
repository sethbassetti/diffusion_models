import math
import random

import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from mnist_data import MNISTDataset
from diffusion import GaussianDiffusion
from model import UNet



# This is the amount of timesteps we can sample from
T = 1000

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

def reverse_transform(image):
    """ Takes in a tensor image and converts it to a uint8 numpy array"""

    image = (image + 1) / 2                 # Range [-1, 1] -> [0, 1)
    image *= 255                            # Range [0, 1) -> [0, 255)
    image = image.numpy().astype(np.uint8)  # Cast image to numpy and make it an unsigned integer type (no negatives)

    return image

@torch.no_grad()
def reverse_diff(model, diffuser, shape):
    """Constructs a sequence of frames of the denoising process"""
    device = next(model.parameters()).device

    b = shape[0]

    # Start imgs out with random noise
    img = torch.randn(shape, device=device)

    imgs = []
    
    # Loops through each timestep to get to the generated image
    for i in tqdm(reversed(range(0, T)), desc='sampling loop time step', total=T):

        # Samples from the specific timestep
        img = diffuser.p_sample(model, img, torch.full((b,), i, device=device), i)
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
    train_model()

def train_model():

    # Define Hyperparameters
    batch_size = 128
    epochs = 100
    lr = 2e-4
    device = 0
    channel_space = 64

    image_size = 28
    image_channels = 1
    dim_mults = (1, 2)
    vartype = 'learned'
    n_log_images = 9        # How many images to log to wandb each epoch

    model_checkpoint = None

    diffuser = GaussianDiffusion(linear_schedule(T), vartype, T)

    # Define the dataset and dataloader
    train_set = MNISTDataset()
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)

    # Define model, optimizer, and loss function
    model = UNet(img_start_channels = image_channels, channel_space=channel_space, dim_mults=dim_mults, 
    vartype=vartype).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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

            loss = diffuser.compute_losses(model, images, device)

            # Apply loss to each prediction and update count
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

        real_img_grid = make_grid(real_images, nrow=3, normalize=True)

        # Generate a batch of images
        gen_imgs = reverse_diff(model, diffuser, (n_log_images, image_channels, image_size, image_size))

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


