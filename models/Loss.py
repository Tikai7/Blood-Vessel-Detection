import torch.nn as nn

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
    def bce_loss(y_pred, y_true):
        """Method to compute the binary cross-entropy loss between the predicted and true labels.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """

        loss = nn.BCEWithLogitsLoss()
        return loss(y_pred, y_true)
    
    @staticmethod
    def dice_loss(y_pred, y_true):
        """Method to compute the Dice loss between the predicted and true labels.
        The method uses the sigmoid function to convert the predicted labels to probabilities.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """
        sigmoid = nn.Sigmoid()
        y_pred = sigmoid(y_pred)
        intersection = (y_pred * y_true).sum()
        return 1 - (2. * intersection + SMOOTH) / (y_pred.sum() + y_true.sum() + SMOOTH)
    
    @staticmethod
    def combined_loss(y_pred, y_true):
        """Method to compute the combined loss, which is the sum of the binary cross-entropy loss and the Dice loss.
        @param y_pred : torch.Tensor, The predicted labels.
        @param y_true : torch.Tensor, The true labels.
        """
        return Loss.bce_loss(y_pred, y_true) + Loss.dice_loss(y_pred, y_true)