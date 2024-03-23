import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from DataLoader import DataLoaderManager
from Train import Trainer
from Loss import Loss
import segmentation_models_pytorch as smp


# ------------------------------ UTILS ------------------------------
def calculate_weights(dataloader):
    n_negative = 0
    n_positive = 0
    for _, mask in dataloader:
        n_negative += (mask == 0).sum().item()
        n_positive += (mask == 1).sum().item()
    pos_weight = n_negative/n_positive
    return pos_weight

# ------------------------------ CONSTANTS ------------------------------
epochs = 100
image_size = 224
num_classes = 1
learning_rate = 1e-4
train_size = 0.7
test_size = 0.15
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ------------------------------ DATA LOADING ------------------------------
dataset = DataLoaderManager(root_dir="patches",kidney_dir=False, data_augmentation=True, shape=(image_size,image_size))

train_size = int(train_size * len(dataset))
test_size =  int(test_size * len(dataset))
val_size = len(dataset) - train_size - test_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

dataset.show_data(train_loader)
dataset.show_data(val_loader)
dataset.show_data(test_loader) 

print(f"Train size : {len(train_loader)}")
print(f"Validation size : {len(val_loader)}")
print(f"Test size : {len(test_loader)}")
# ------------------------------ MODEL TRAINING ------------------------------
# Encoder can be one of 'vgg16', 'vgg19', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'efficientnet-b3'
encoder_name = 'efficientnet-b3'
model = smp.Unet(encoder_name=encoder_name, encoder_weights='imagenet', classes=num_classes, in_channels=3)
model = model.to(device)
optimizer = torch.optim.AdamW
loss = Loss.combined_loss
trainer = Trainer()
pos_weight = trainer.calculate_weights(train_loader)
print(f"Positive weights : {pos_weight}")
train_loss, val_loss, precision, recall, val_p, val_r = trainer.set_model(model)\
    .set_loader(train_loader, val_loader, test_loader)\
    .set_loss_fn(loss)\
    .set_optimizer(optimizer)\
    .fit(learning_rate, epochs, 1/pos_weight)
# ------------------------------ PLOTTING ------------------------------
trainer.plot_loss(train_loss, val_loss)
trainer.plot_precision_recall(precision, recall, val_p, val_r)
trainer.evaluate()
torch.save(model.state_dict(), "params/local_model_EN_b3_BD_adamW_weight_augmented_3D_3C_100epochs")
