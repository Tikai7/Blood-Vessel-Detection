import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

class Tester():
    """A class to represent the testing process for the U-Net model for vessel segmentation.
    """

    def __init__(self, state="res/model_params/model_EN_b3_BD_adamW_weight_augmented_3D_3C_100epochs") -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state = torch.load(state)
        encoder_name = 'efficientnet-b3'
        model = smp.Unet(encoder_name=encoder_name, classes=1, in_channels=3)
        model.load_state_dict(self.state)
        model.eval()
        self.model = model
        self.model = self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def minmax(self, image):
        """Method to normalize the image.
        @param image, The image to be normalized.
        """
        image = (image - image.min()) / (image.max() - image.min())
        return image
    
    def binarize(self, mask):
        """Method to binarize the mask.
        @param mask, The mask to be binarized.
        """
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return mask
    
    def predict(self, image):
        """Method to predict the mask for the input image.
        @param image, The input image.
        """
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        mask = self.model(image)        
        mask = mask.squeeze(0)
        mask = mask.detach().cpu().numpy()
        mask = self.binarize(mask)
        image = image.squeeze(0)
        image = image.detach().cpu().numpy()
        masked_image = image*mask
        mask_image_transposed = np.transpose(masked_image, (1, 2, 0))
        return mask, mask_image_transposed
    
    def predict_loader(self, loader, max_images=5, has_target=True):
        """Method to predict the mask for the input data loader.
        @param loader, The input data loader.
        """
        for i, batch in enumerate(loader):
            
            if has_target:
                image, real_mask = batch
            else:
                image = batch

            image = image.to(self.device)
            
            mask = self.model(image)        
            mask = mask.squeeze(0)
            mask = mask.detach().cpu().numpy()
            mask = self.binarize(mask)

            image = image.squeeze(0)
            image = image.detach().cpu().numpy()
            masked_image = image*mask

            if has_target:
                real_mask = real_mask.squeeze(0)
                real_mask = real_mask.detach().cpu().numpy()
                real_mask = self.binarize(real_mask)
                real_mask = np.transpose(real_mask[0], (1, 2, 0))

            image_transposed = np.transpose(image[0], (1, 2, 0))
            mask_transposed = np.transpose(mask[0], (1, 2, 0))
            masked_image = np.transpose(masked_image[0], (1, 2, 0))

            nb_subplots = 4 if has_target else 3   

            if i < max_images:
                plt.figure(figsize=(15,10))
                plt.subplot(1, nb_subplots, 1)
                plt.imshow(image_transposed)
                plt.title("Input Image")
                plt.subplot(1, nb_subplots, 2)
                plt.imshow(real_mask)
                plt.title("Real Mask")
                plt.subplot(1, nb_subplots, 3)
                plt.imshow(mask_transposed)
                plt.title("Predicted Mask")
                if has_target:
                    plt.subplot(1, nb_subplots, 4)
                    plt.imshow(masked_image)
                    plt.title("Masked Image")
                plt.show()
            else:
                break

