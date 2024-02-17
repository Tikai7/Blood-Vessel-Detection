import cv2
from image_processing import Processing
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import  DataLoader, random_split
from torchvision import transforms
from sklearn.decomposition import PCA

from models.DataLoader import DataLoaderManager
from models.Loss import Loss
from models.Train import Trainer
from models.Unet import VesselsModel


# Faire le mask des images
# Virer tout gris / + detecter sang
# LabelMe (manuellement) 
# Check eosin component 
# Se focus sur le sang #ff8001
# numéro cluster / ligne colonne

def main():
    processing = Processing()
    image_patch = cv2.imread("images/4_618_001B._patches_18_14.png")
    hematoxylin, eosin, mask, vessels_count = processing.process_patch(image_patch, use_hematoxylin=False, min_size=100, get_vessel_count=True)
    processing.visualize_results()


# def start():
#     dataset = DataLoaderManager("dataset/archive", (64,64))

#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size

#     train_set, val_set = random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

#     dataset.show_data(train_loader)
#     dataset.show_data(val_loader)

#     model = VesselsModel(1)
#     loss_fn = Loss.combined_loss
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     trainer = Trainer() \
#         .set_model(model) \
#         .set_loader(train_loader, val_loader) \
#         .set_loss_fn(loss_fn) \
#         .set_optimizer(optimizer) \

#     train_loss, val_loss = trainer.fit(epochs=1)

#     plt.style.use("ggplot")
#     plt.figure()
#     plt.plot(train_loss, label="Training Loss")
#     plt.plot(val_loss, label="Validation Loss")
#     plt.legend()
#     plt.show()


if __name__ == "__main__":
    main()
    # start()