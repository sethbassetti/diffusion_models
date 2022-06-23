from torchvision import datasets, transforms
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
        
        # Randomly flip half of the images horizontally
        transform = transforms.RandomHorizontalFlip()

        # Loads the dataset, downloading if it is not already downloaded and converts it into pytorch
        dataset = torch.tensor(datasets.FashionMNIST('./data/', train=True, download=True).data)

        # If there is a channel dimension then reshape from N x H x W x C -> N x C x H x W
        if len(dataset.shape) > 3:
            dataset = dataset.permute(0, 3, 1, 2)
        
        # Otherwise add a channel dimension
        else:
            dataset = dataset.unsqueeze(1)

        # Converts images from [0, 255) integer scale to [0,1) float scale
        images = dataset / 255

        # Standardizes image values to be between 0 and 1
        images = images * 2 - 1

        # Adds a channel dimension to the images and applies the transform to them
        images = transform(images)

        return images

    def __getitem__(self, idx):

        # Grabs an image from the data
        image = self.images[idx]

        # Grabs a random timestep from 1 to the total number of timesteps 
        timestep = torch.randint(1, self.max_timestep, (1,))

        return image, timestep

    def __len__(self):
        return len(self.images)

