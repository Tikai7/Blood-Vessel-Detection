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

    def __init__(self, root_dir, kidney_dir=False, shape=(64,64)) -> None:
        self.SHAPE = shape
        self.root_dir = root_dir
        self.kidney_dir = "dataset/archive"

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
        print(xi.shape, yi.shape)
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title('Image')
        plt.imshow(xi[0][0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Mask')
        plt.imshow(yi[0][0], cmap='gray')
        plt.show()

        