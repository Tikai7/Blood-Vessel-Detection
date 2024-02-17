import torch
from torch.utils.data import DataLoader

class Trainer:
    """A class to represent the training process for the U-Net model for vessel segmentation.
    - The class contains a fit method to train the model.
    - The fit method takes the model, training and validation data loaders, loss function, optimizer, device, and number of epochs as input.
    - The fit method trains the model for the specified number of epochs and returns the training and validation losses.
    """
    def __init__(self) -> None:
        self.model : torch.nn.Module = None
        self.train_loader : DataLoader = None
        self.val_loader : DataLoader = None
        self.loss_fn : torch.nn.Module = None
        self.optimizer : torch.optim.Optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._OPT_LIST = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamW" : torch.optim.AdamW
        }

    def set_optimizer(self, optimizer : str):
        """Method to set the optimizer for the model.
        @param optimizer, The optimizer to be used for training the model.
        """
        self.optimizer : torch.optim.Optimizer = optimizer
        return self
    
    def set_model(self, model : torch.nn.Module):
        """Method to set the model for training.
        @param model, The model to be trained.
        """
        self.model = model
        return self
    
    def set_loader(self, train_loader : DataLoader, val_loader : DataLoader):
        """Method to set the training and validation data loaders.
        @param train_loader : DataLoader, The training data loader.
        @param val_loader : DataLoader, The validation data loader.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        return self
    
    def set_loss_fn(self, loss_fn : torch.nn.Module):
        """Method to set the loss function for training the model.
        @param loss_fn, The loss function to be used for training the model.
        """
        self.loss_fn = loss_fn
        return self
    

    def fit(self, learning_rate : float = 1e-4, epochs : int = 100):
        """Method to train the model.
        @param learning_rate : float, The learning rate for the optimizer.
        @param epochs : int, The number of epochs for training the model.
        """
        
        val_loss, train_loss = [], []
        self.model = self.model.to(self.device)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            # Training
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(batch_x)
                loss = self.loss_fn(y_pred, batch_y)
                loss.backward()
                loss_item = loss.item()
                train_loss.append(loss_item )
                self.optimizer.step()
            # Validation
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                with torch.no_grad():
                    y_pred = self.model(batch_x)
                    val_loss = self.loss_fn(y_pred, batch_y)
                    val_loss.append(val_loss.item())

        return train_loss, val_loss
    


