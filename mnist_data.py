from torchvision import datasets
from torch.utils.data import Dataset
import torch

class MNISTDataset(Dataset):
    """Dataset class that holds MNIST images for diffusion model training"""

    def __init__(self, T):
        super().__init__()

        # Loads the data into memory
        self.images = self.load_data()

        # Used to randomly select a timestep when grabbing image from the dataset
        self.max_timestep = T

    def load_data(self):
        """Performs the loading of the data and all normalization/standardization"""
        
        # Loads the MNIST dataset, downloading if it is not already downloaded
        dataset = datasets.MNIST('./data/', train=True, download=True)

        # Converts images from [0, 255) integer scale to [0,1) float scale
        images = dataset.data / 255

        # Standardizes image values to be between 0 and 1
        images = images * 2 - 1

        # Adds a channel dimension to the images
        images = images.unsqueeze(1)

        return images

    def __getitem__(self, idx):

        # Grabs an image from the data
        image = self.images[idx]

        # Grabs a random timestep from 1 to the total number of timesteps 
        timestep = torch.randint(1, self.max_timestep, (1,))

        return image, timestep

    def __len__(self):
        return len(self.images)

