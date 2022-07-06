import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import numpy as np

class CelebDataset(Dataset):
    """Dataset class that holds CelebHQ images for diffusion model training"""

    def __init__(self, data_path):
        super().__init__()

        # Loads the data into memory
        self.images = self.load_data(data_path)

    def load_data(self, data_path):
        """Performs the loading of the data and all normalization/standardization"""
        
        # Traverse through the image directory and read each image into a tensor
        image_list = [read_image(os.path.join(data_path, file)) for file in os.listdir(data_path)]
        images = torch.stack(image_list)

        # Converts images from [0, 255) integer scale to [0,1) float scale
        images = images / 255

        # Standardizes image values to be between 0 and 1
        images = images * 2 - 1

        # Randomly flip half of the images horizontally
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64))]
        )

        # Adds a channel dimension to the images and applies the transform to them
        images = transform(images)

        return images

    def __getitem__(self, idx):

        # Grabs an image from the data
        image = self.images[idx]

        return image

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":

    # Loads up the data into a dataset
    data_path = "/home/bassets/diffusion_models/data/celebHQ/"
    celeb_data = CelebDataset(data_path)

    # Takes first examples and go from [-1, 1] -> [0, 1)
    examples = celeb_data[:9]
    examples = (examples + 1) / 2

    # Make a grid of the examples, move channel dimension and display it
    grid = make_grid(examples, nrow=3)
    grid = grid.permute(1, 2, 0).numpy()
    plt.imshow(grid)