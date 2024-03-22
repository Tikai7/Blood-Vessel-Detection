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
    def dice_loss(y_pred, y_true, pos_weight=None):
        """Method to compute the Dice loss between the predicted and true labels.
        The method uses the sigmoid function to convert the predicted labels to probabilities.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """
        dice_loss = smp.losses.DiceLoss(mode='binary')
        return dice_loss(y_pred,y_true)

    @staticmethod
    def combined_loss(y_pred, y_true, pos_weight=None):
        """Method to compute the combined loss, which is the sum of the binary cross-entropy loss and the Dice loss.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """
        return 0.5*(Loss.bce_loss(y_pred, y_true, pos_weight) + Loss.dice_loss(y_pred, y_true))