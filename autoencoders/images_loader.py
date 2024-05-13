import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.load_images_into_memory(directory)

    def load_images_into_memory(self, directory):
        for fname in os.listdir(directory):
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(directory, fname)
                image = Image.open(path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
