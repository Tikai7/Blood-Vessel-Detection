import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

class Tester():
    """A class to represent the testing process for the U-Net model for vessel segmentation.
    """

    def __init__(self, state="res/model_params/model_ENB3_BCE_DICE_balanced_augmented_3D", test_loader=None, encoder_name='efficientnet-b3') -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_loader = test_loader
        self.state = torch.load(state)
        encoder_name = encoder_name
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
        @return image, The normalized image.
        """
        image = (image - image.min()) / (image.max() - image.min())
        return image
    
    def binarize(self, mask):
        """Method to binarize the mask.
        @param mask, The mask to be binarized.
        @return mask, The binarized mask.
        """
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return mask
    
    def predict(self, image):
        """Method to predict the mask for the input image.
        @param image, The input image.
        @return mask, The masked image.
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
                # if has_target:
                #     plt.subplot(1, nb_subplots, 4)
                #     plt.imshow(masked_image)
                #     plt.title("Masked Image")
                plt.show()
            else:
                break
    

    def calculate_accuracy(self, y_true, y_pred): 
        """Calculate the accuracy of the model.
        @param y_true : torch.Tensor, The ground truth binary masks.
        @param y_pred : torch.Tensor, The predicted binary masks.
        """
        y_pred_bin = (y_pred > 0.5).float()
        accuracy = (y_pred_bin == y_true).sum().item() / (y_true.shape[0] * y_true.shape[1] * y_true.shape[2] * y_true.shape[3])
        return accuracy
    

    def calculate_precision_recall(self, y_true, y_pred):
        """Calculate precision and recall.
        @param y_true : torch.Tensor, The ground truth binary masks.
        @param y_pred : torch.Tensor, The predicted binary masks.
        """
        # Threshold predictions to binary
        y_pred_bin = (y_pred > 0.5).float()

        # Calculate true positives, false positives, and false negatives
        tp = ((y_true * y_pred_bin).sum().item())
        fp = ((y_pred_bin - y_true).clamp(min=0).sum().item())
        fn = ((y_true - y_pred_bin).clamp(min=0).sum().item())

        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        return precision, recall
    
    def evaluate(self):
        """Method to test the model on a batch of validation/test data.
        @return accuracy : float, The accuracy of the model.
        @return precision : float, The precision of the model.
        @return recall : float, The recall of the model.
        @return f1_score : float, The F1 score of the model.
        """
        print("Testing the model")
        if self.test_loader is None:
            raise ValueError("Test loader is not provided")
        y_true_all = []
        y_pred_all = []
        precision, recall = 0, 0
        f1_score, accuracy = 0, 0
        self.model.eval()
        self.model.to(self.device)
        for batch_x, batch_y in self.test_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            mask = self.model(batch_x)
            p, r = self.calculate_precision_recall(batch_y, mask)
            precision += p
            recall += r
            y_pred_all.append(mask.detach().cpu())
            y_true_all.append(batch_y.cpu())
            
        y_pred_all = torch.cat(y_pred_all, dim=0)
        y_true_all = torch.cat(y_true_all, dim=0)
        accuracy = self.calculate_accuracy(y_true_all, y_pred_all)
        precision = precision/len(self.test_loader)
        recall = recall/len(self.test_loader)
        f1_score = 2*precision*recall/(precision+recall)

        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
    
        return accuracy, precision, recall, f1_score
