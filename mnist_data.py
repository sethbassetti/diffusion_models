from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch

class MNISTDataset(Dataset):
    """Dataset class that holds MNIST images for diffusion model training"""

    def __init__(self):
        super().__init__()

        # Loads the data into memory
        self.images, self.labels = self.load_data()

    def load_data(self):
        """Performs the loading of the data and all normalization/standardization"""
        
        # Randomly flip half of the images horizontally
        transform = transforms.RandomHorizontalFlip()

        # Loads the dataset, downloading if it is not already downloaded and converts it into pytorch
        dataset = datasets.FashionMNIST('./data/', train=True, download=True)
        images = dataset.data
        labels = dataset.targets

        # If there is a channel dimension then reshape from N x H x W x C -> N x C x H x W
        if len(images.shape) > 3:
            images = images.permute(0, 3, 1, 2)
        
        # Otherwise add a channel dimension
        else:
            images = images.unsqueeze(1)

        # Converts images from [0, 255) integer scale to [0,1) float scale
        images = images / 255

        # Standardizes image values to be between 0 and 1
        images = images * 2 - 1

        # Adds a channel dimension to the images and applies the transform to them
        images = transform(images)

        return images, labels

    def __getitem__(self, idx):

        # Grabs an image from the data
        image = self.images[idx]
        label = self.labels[idx]

        return image, label

    def __len__(self):
        return len(self.images)

