import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import  DataLoader, random_split
from torchvision import transforms
from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class DataLoaderManager(Dataset):
    """A class to represent the data loader manager for the U-Net model for vessel segmentation.
    - The class contains static methods to load and preprocess the data.
    - The class contains a static method to visualize the data.
    - The class contains a static method to build a data loader from the input data.
    """

    def __init__(self, root_dir, shape=(64,64)) -> None:
        self.SHAPE = shape
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'img')
        self.mask_folder = os.path.join(root_dir, 'mask')
        self.image_filenames = os.listdir(self.image_folder)
        self.mask_filenames = os.listdir(self.mask_folder)
        self.transform = transforms.Compose([
            transforms.Resize(self.SHAPE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.target_transform = transforms.Compose([
            transforms.Resize(self.SHAPE),
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_folder, self.mask_filenames[idx])

        image = Image.open(img_name).convert('L')        
        mask = Image.open(mask_name).convert('L')  
    
        image = self.transform(image)
        mask = self.target_transform(mask)

        return image, mask
    
    def show_data(self, X):
        """A static method to visualize the input data.
        @param X: The input data.
        @param shape: The shape of the input data.
        """
        
        xi, yi = next(iter(X))
        print(xi.shape, yi.shape)
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title('Image')
        plt.imshow(xi[0][0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Mask')
        plt.imshow(yi[0][0], cmap='gray')
        plt.show()

# dataset = DataLoaderManager("dataset/archive", (64,64))

# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size

# train_set, val_set = random_split(dataset, [train_size, val_size])

# train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

# dataset.show_data(train_loader)
