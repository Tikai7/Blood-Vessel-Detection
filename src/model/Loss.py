import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Smoothing factor for the Dice loss
SMOOTH = 1e-6

class Loss():
    """A class to represent the loss functions used for training the U-Net model for vessel segmentation.
    - The class contains two static methods: bce_loss and dice_loss.
    - The bce_loss method computes the binary cross-entropy loss between the predicted and true labels.
    - The dice_loss method computes the Dice loss between the predicted and true labels.
    - The combined_loss method computes the combined loss, which is the sum of the binary cross-entropy loss and the Dice loss.
    """
    @staticmethod
    def bce_loss(y_pred, y_true, pos_weight=None):
        """Method to compute the binary cross-entropy loss between the predicted and true labels.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """
        pos_weight = torch.tensor([pos_weight]).to(y_true.device) if pos_weight else None
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss(y_pred,  y_true)

    # twersky et FOCAL LOSS
    @staticmethod
    def focal_loss(y_pred, y_true, pos_weight=None):
        """Method to compute the Focal loss between the predicted and true labels.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """
        focal_loss = smp.losses.FocalLoss(alpha = pos_weight,gamma=4,mode='binary')
        return focal_loss(y_pred,y_true,pos_weight)
    
    @staticmethod
    def tversky_loss(y_pred, y_true, pos_weight=None):
        """Method to compute the Tversky loss between the predicted and true labels.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """
        tversky_loss = smp.losses.TverskyLoss(pos_weight, 1-pos_weight,mode='binary')
        return tversky_loss(y_pred,y_true,pos_weight)
        
    @staticmethod
    def dice_loss(y_pred, y_true, pos_weight=None):
        """Method to compute the Dice loss between the predicted and true labels.
        The method uses the sigmoid function to convert the predicted labels to probabilities.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """
        dice_loss = smp.losses.DiceLoss(mode='binary')
        return dice_loss(y_pred,y_true)

    @staticmethod
    def combined_bce_loss(y_pred, y_true, pos_weight=None):
        """Method to compute the combined loss, which is the sum of the binary cross-entropy loss and the Dice loss.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """
        return 0.5*(Loss.bce_loss(y_pred, y_true, pos_weight) + Loss.dice_loss(y_pred, y_true))
    
    @staticmethod
    def combined_focal_loss(y_pred, y_true, pos_weight=None):
        """Method to compute the combined loss, which is the sum of the binary cross-entropy loss and the Dice loss.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """
        return 0.5*(Loss.focal_loss(y_pred, y_true, pos_weight) + Loss.dice_loss(y_pred, y_true))
    
    @staticmethod
    def combined_tversky_loss(y_pred, y_true, pos_weight=None):
        """Method to compute the combined loss, which is the sum of the binary cross-entropy loss and the Dice loss.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """
        return 0.5*(Loss.tversky_loss(y_pred, y_true, pos_weight) + Loss.dice_loss(y_pred, y_true))