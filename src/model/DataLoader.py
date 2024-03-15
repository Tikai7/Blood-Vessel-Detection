import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

def binarize(image):
    image[image > 0] = 1
    return image

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
            transforms.ToTensor(),
            transforms.Lambda(binarize), 
        ])

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_folder, self.mask_filenames[idx])
        try:
            image = Image.open(img_name).convert('L')
            mask = Image.open(mask_name).convert('L')
        except:
#             print(f"Unable to open image file {img_name}. Skipping this file.")
            return self.__getitem__(idx + 1)  # try the next file

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

        