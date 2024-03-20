import torch
import matplotlib.pyplot as plt
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
    
    def plot_precision_recall(self, precision : list, recall : list, val_precision : list, val_recall : list):
        """Method to plot the precision and recall.
        @param precision : list, The list of precision values for the training data.
        @param recall : list, The list of recall values for the training data.
        @param val_precision : list, The list of precision values for the validation data.
        @param val_recall : list, The list of recall values for the validation data.
        """
        plt.style.use('ggplot')
        plt.figure(figsize=(12,7))
        plt.plot(precision, label='precision')
        plt.plot(recall, label='recall')
        plt.plot(val_precision, label='val precision')
        plt.plot(val_recall, label='val recall')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Training and Validation Precision and Recall')
        plt.show()

    def plot_loss(self, train_loss : list, val_loss : list):
        """Method to plot the training and validation loss.
        @param train_loss : list, The list of training losses.
        @param val_loss : list, The list of validation losses.
        """
        plt.style.use('ggplot')
        plt.figure(figsize=(15,10))
        plt.plot(train_loss, label='train loss')
        plt.plot(val_loss, label='val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def calculate_weights(self, dataloader):
        n_negative = 0
        n_positive = 0
        for _, mask in dataloader:
            n_negative += (mask == 0).sum().item()
            n_positive += (mask == 1).sum().item()
        pos_weight = n_negative/n_positive
        return pos_weight

    def test_on_batch(self, loader : DataLoader):
        """Method to test the model on a batch of validation/test data.
        @param loader : DataLoader, The validation or test data loader.
        """
        print("Testing the model on a batch of validation data...")
        self.model.eval()
        i,(batch_x,batch_y) = next(enumerate(loader))
        self.model = self.model.to(self.device)
        batch_x = batch_x.to(self.device)
        mask = self.model.predict(batch_x)
        mask = mask.cpu().numpy()
        batch_x = batch_x.cpu().numpy()

        mask = (mask > 0.5)

        plt.figure(figsize=(12,7))
        plt.subplot(131)
        plt.imshow(batch_x[0][0,:,:])
        plt.subplot(132)
        plt.imshow(batch_y[0][0,:,:])
        plt.subplot(133)
        plt.imshow(mask[0][0,:,:])
        plt.show()

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
    
    def fit(self, learning_rate = 1e-4, epochs : int = 100, pos_weight : float = None):
        """Method to train the model.
        @param learning_rate : float, The learning rate for the optimizer.
        @param epochs : int, The number of epochs for training the model.
        """
        print(f"Training the model on {self.device}...")
        self.model.to(self.device)
        self.optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        val_loss, train_loss = [], []
        val_precision, val_recall = [], []
        precision, recall = [], []
        for epoch in range(epochs):
            # Training
            train_epoch_loss = 0
            train_precision, train_recall = 0, 0

            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                y_pred = self.model(batch_x)
                with torch.no_grad():
                    p, r = self.calculate_precision_recall(batch_y, y_pred)
                    train_precision += p
                    train_recall += r
                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, batch_y, pos_weight)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss += loss.item()
            train_loss.append(train_epoch_loss/len(self.train_loader))
            precision.append(train_precision/len(self.train_loader))
            recall.append(train_recall/len(self.train_loader))
            # Validation
            print("Validating...")
            with torch.no_grad():
                val_epoch_loss = 0
                val_p, val_r = 0, 0
                for batch_x, batch_y in self.val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    y_pred = self.model(batch_x)
                    p, r = self.calculate_precision_recall(batch_y, y_pred)
                    val_p += p
                    val_r += r
                    loss_val = self.loss_fn(y_pred, batch_y, pos_weight)
                    val_epoch_loss += loss_val.item()
                    
                val_loss.append(val_epoch_loss / len(self.val_loader))
                val_precision.append(val_p/len(self.val_loader))
                val_recall.append(val_r/len(self.val_loader))
            
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss[-1]}, Validation Loss: {val_loss[-1]}")

        print("Training complete.")
        return train_loss, val_loss, precision, recall, val_precision, val_recall 