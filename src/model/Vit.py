import torch
from torch.utils.data import DataLoader
from vit_pytorch import ViT
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from DataLoader import DataLoaderManager
from Train import Trainer
from Loss import Loss
import segmentation_models_pytorch as smp


# ------------------------------ CONSTANTS ------------------------------
image_size = 224
patch_size = 16
num_classes = 2  
learning_rate = 1e-4
epochs = 1
train_size = 0.8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------ DATA LOADING ------------------------------
dataset = DataLoaderManager("patches", shape=(image_size,image_size))


train_size = int(train_size * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

# dataset.show_data(train_loader)
# dataset.show_data(val_loader)

# ------------------------------ MODEL TRAINING ------------------------------
# Encoder can be one of 'vgg16', 'vgg19', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d'
encoder_name = 'vgg16'
backbone = smp.encoders.get_encoder(encoder_name, pretrained=True)
model = smp.Unet(encoder_name=encoder_name, encoder_weights='imagenet', classes=1, in_channels=1)
model.to(device)
optimizer = torch.optim.Adam
loss = Loss.bce_loss
trainer = Trainer()
train_loss, val_loss = trainer.set_model(model)\
    .set_loader(train_loader, val_loader)\
    .set_loss_fn(loss)\
    .set_optimizer(optimizer)\
    .fit(learning_rate, epochs)
# ------------------------------ PLOTTING ------------------------------
trainer.plot_loss(train_loss, val_loss)

