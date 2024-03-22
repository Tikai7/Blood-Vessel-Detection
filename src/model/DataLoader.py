import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from Processing import Processing
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import os

class DataLoaderManager(Dataset):
    """A class to represent the data loader manager for the U-Net model for vessel segmentation.
    - The class contains static methods to load and preprocess the data.
    - The class contains a static method to visualize the data.
    - The class contains a static method to build a data loader from the input data.
    """

    def __init__(self, root_dir, kidney_dir=False, data_augmentation=False, shape=(64,64)) -> None:
        self.SHAPE = shape
        self.root_dir = root_dir
        self.kidney_dir = "dataset/archive"
        self.data_augmentation = data_augmentation

        if kidney_dir:
            self.image_folder = os.path.join(self.kidney_dir, 'img')
            self.mask_folder = os.path.join(self.kidney_dir, 'mask')
            self.mask_filenames = os.listdir(self.image_folder)
            self.image_filenames = os.listdir(self.mask_folder)
        else:
            self.image_folder = os.path.join(root_dir, 'patches_bvd_clustd')
            self.mask_folder = os.path.join(root_dir, 'patches_bvd_clustd_mask')
            self.mask_filenames = self.get_filenames(self.mask_folder)
            self.image_filenames = self.get_filenames(self.image_folder, self.mask_filenames)

        self.same_transform = transforms.Compose([
            transforms.Resize(self.SHAPE),
            transforms.ToTensor(),
        ])
        self.transform = transforms.Compose([
            self.same_transform,
            transforms.Normalize(mean=[0.5], std=[0.5]),
            # transforms.Lambda(self.minmax),
        ])
        self.target_transform = transforms.Compose([
            self.same_transform,
            transforms.Lambda(self.binarize), 
        ])

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_folder, self.mask_filenames[idx])
        try:
            image = Image.open(img_name)
            mask = Image.open(mask_name).convert('L')
        except:
            return self.__getitem__(idx + 1)  


        if self.data_augmentation:
            image, mask = self.data_augmentation_transform(image, mask)
            
        image = self.transform(image)
        mask = self.target_transform(mask)
        
        return image, mask
    
    def data_augmentation_transform(self, image, mask):
        """A static method to perform data augmentation on the input image.
        @param image: The input image.
        @return: The augmented image.
        """ 
        if torch.rand(1) > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        if torch.rand(1) > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
        if torch.rand(1) > 0.5:
            angle = torch.randint(low=-10, high=10, size=(1,)).item()
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)
        
        return image, mask
    
    def minmax(self, image):
        image = (image - image.min()) / (image.max() - image.min())
        return image
    
    def binarize(self, image):
        image[image > 0] = 1
        return image
    
    def get_filenames(self, folder, mask_filenames=None):
        clusters = os.listdir(folder)
        filenames = []
        for cluster in clusters:
            files = os.listdir(os.path.join(folder, cluster))
            files = [os.path.join(cluster, file) for file in files]
            filenames.extend(files)
        if mask_filenames:
            filenames = [filename for filename in filenames if filename in mask_filenames]
        return filenames

    def show_data(self, X):
        """A static method to visualize the input data.
        @param X: The input data.
        @param shape: The shape of the input data.
        """
        
        xi, yi = next(iter(X))
        print("Shape of the input data: ")
        print(xi.shape, yi.shape)
        print("Min and Max of the input data: ")
        print(xi.min(), xi.max(), yi.min(), yi.max())
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title('Image')
        plt.imshow(xi[0][0])
        plt.subplot(1, 2, 2)
        plt.title('Mask')
        plt.imshow(yi[0][0], cmap='gray')
        plt.show()

        