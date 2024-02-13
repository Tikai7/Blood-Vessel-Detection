import torch

class Trainer:
    """A class to represent the training process for the U-Net model for vessel segmentation.
    - The class contains a fit method to train the model.
    - The fit method takes the model, training and validation data loaders, loss function, optimizer, device, and number of epochs as input.
    - The fit method trains the model for the specified number of epochs and returns the training and validation losses.
    """
    def __init__(self) -> None:
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.loss_fn = None
        self.optimizer = None
        self.device = None
    
    def fit(self, model, train_loader, val_loader, loss_fn, optimizer, device, epochs=100):
        val_loss, train_loss = [], []
        
        model = model.to(device)
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            # Training
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                y_pred = model(batch_x)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                loss_item = loss.item()
                train_loss.append(loss_item )
                optimizer.step()
            # Validation
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                with torch.no_grad():
                    y_pred = model(batch_x)
                    val_loss = loss_fn(y_pred, batch_y)
                    val_loss.append(val_loss.item())

        return train_loss, val_loss