import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class Trainer:
    """A class to represent the training process for the U-Net model for vessel segmentation.
    """
    def __init__(self) -> None:
        self.model : torch.nn.Module = None
        self.train_loader : DataLoader = None
        self.val_loader : DataLoader = None
        self.loss_fn : torch.nn.Module = None
        self.optimizer : torch.optim.Optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {
            "validation": {
                "accuracy": [],
                "loss": [],
                "precision": [],
                "recall": []
            },
            "training": {
                "accuracy": [],
                "loss": [],
                "precision": [],
                "recall": []
            },
            "params": {
                "learning_rate": None,
                "pos_weight": None,
                "weight_decay": None,
                "epochs": None
            }
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
    
    def set_loader(self, train_loader : DataLoader, val_loader : DataLoader, test_loader : DataLoader):
        """Method to set the training and validation data loaders.
        @param train_loader : DataLoader, The training data loader.
        @param val_loader : DataLoader, The validation data loader.
        @param test_loader : DataLoader, The test data loader.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
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
        plt.plot(precision, label='train precision')
        plt.plot(recall, label='train recall')
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
        """Method to calculate the positive weights for the loss function.
        @param dataloader : DataLoader, The data loader to calculate the positive weights.
        """
        n_negative = 0
        n_positive = 0
        for _, mask in dataloader:
            n_negative += (mask == 0).sum().item()
            n_positive += (mask == 1).sum().item()
        pos_weight = n_negative/n_positive
        return pos_weight

    def calculate_proba(self, dataloader):
        n_negative = 0
        n_positive = 0
        for _, mask in dataloader:
            n_negative += (mask == 0).sum().item()
            n_positive += (mask == 1).sum().item()
        pos_proba = n_positive/(n_positive+n_negative)
        return pos_proba

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


    def save(self, path : str = "model.pth", history_path : str = "history.txt"):
        """Method to save the model.
        """
        print("Saving the model...")
        torch.save(self.model.state_dict(), path)
        torch.save(self.history, history_path)
        print("Model saved.")


    def fit(self, learning_rate = 1e-4, epochs : int = 100, pos_proba : float = None, pos_weight : float = None, weight_decay : float = 0.01):
        """Method to train the model.
        @param learning_rate : float, The learning rate for the optimizer.
        @param epochs : int, The number of epochs for training the model.
        """
        print(f"Training the model on {self.device}...")
        self.model.to(self.device)
        self.optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.history["params"]["learning_rate"] = learning_rate
        self.history["params"]["pos_weight"] = pos_weight
        self.history["params"]["weight_decay"] = weight_decay
        self.history["params"]["epochs"] = epochs

        val_accuracy, train_accuracy = [], []
        val_loss, train_loss = [], []
        val_precision, train_precision = [], []
        val_recall, train_recall = [], []

        for epoch in range(epochs):
            # ----- Training
            y_true_accuracy, y_pred_accuracy = [], []
            y_true_val_accuracy, y_pred_val_accuracy = [], []

            train_epoch_loss = 0
            precision, recall = 0, 0
            for _ , (batch_x, batch_y) in enumerate(self.train_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                y_pred = self.model(batch_x)
                with torch.no_grad():
                    p, r = self.calculate_precision_recall(batch_y, y_pred)
                    precision += p
                    recall += r
                    y_true_accuracy.append(y_pred.detach().cpu())
                    y_pred_accuracy.append(batch_y.cpu())
            
                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, batch_y, pos_proba, pos_weight)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss += loss.item()

            y_true_accuracy = torch.cat(y_true_accuracy, dim=0)
            y_pred_accuracy = torch.cat(y_pred_accuracy, dim=0)

            train_accuracy.append(self.calculate_accuracy(y_true_accuracy, y_pred_accuracy))
            train_loss.append(train_epoch_loss/len(self.train_loader))
            train_precision.append(precision/len(self.train_loader))
            train_recall.append(recall/len(self.train_loader))

            # ----- Validation
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
                    loss_val = self.loss_fn(y_pred, batch_y, pos_proba, pos_weight)
                    val_epoch_loss += loss_val.item()
                    y_true_val_accuracy.append(y_pred.detach().cpu())
                    y_pred_val_accuracy.append(batch_y.cpu())

                y_true_val_accuracy = torch.cat(y_true_val_accuracy, dim=0)
                y_pred_val_accuracy = torch.cat(y_pred_val_accuracy, dim=0)

                val_accuracy.append(self.calculate_accuracy(y_true_val_accuracy, y_pred_val_accuracy))
                val_loss.append(val_epoch_loss / len(self.val_loader))
                val_precision.append(val_p/len(self.val_loader))
                val_recall.append(val_r/len(self.val_loader))
            
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss[-1]}, Validation Loss: {val_loss[-1]}")

        print("Training complete.")
        
        self.history["training"]["accuracy"] = train_accuracy
        self.history["training"]["loss"] = train_loss
        self.history["training"]["precision"] = train_precision
        self.history["training"]["recall"] = train_recall
        self.history["validation"]["accuracy"] = val_accuracy
        self.history["validation"]["loss"] = val_loss
        self.history["validation"]["precision"] = val_precision
        self.history["validation"]["recall"] = val_recall


        return self.history