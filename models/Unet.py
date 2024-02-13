import torch
import torch.nn as nn

class VesselEncoder(nn.Module):
    """A class to represent the encoder part of the U-Net model for vessel segmentation.
    - The encoder part of the U-Net model is composed of two convolutional layers followed by a batch normalization layer and a ReLU activation function.
    - The encoder part also includes a max pooling layer to downsample the input.
    - @param in_channels: The number of input channels.
    - @param out_channels: The number of output channels.
    """
    def __init__(self,in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding="valid")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding="valid")
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)
        self.downsampling = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.downsampling(x)
        return x

class VesselDecoder(nn.Module):
    """A class to represent the decoder part of the U-Net model for vessel segmentation.
    - The decoder part of the U-Net model is composed of two convolutional layers followed by a batch normalization layer and a ReLU activation function.
    - The decoder part also includes an upsampling layer to upsample the input.
    - @param in_channels: The number of input channels.
    - @param out_channels: The number of output channels.
    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding="valid")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding="valid")
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)
        self.upsampling = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.upsampling(x)
        return x

class VesselsModel(nn.Module):
    """A class to represent the U-Net model for vessel segmentation.
    - The U-Net model is composed of an encoder part and a decoder part.
    - The encoder part is composed of four VesselEncoder objects.
    - The decoder part is composed of four VesselDecoder objects.
    - The model also includes a final convolutional layer and a sigmoid activation function to do binary classification to get the mask.
    - @param in_channels: The number of input channels.
    """

    def __init__(self, in_channels) -> None:
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels,64, kernel_size=(3,3), padding="valid")
        self.encoder = [
            VesselEncoder(64,128),
            VesselEncoder(128,256),
            VesselEncoder(256,512),
            VesselEncoder(512,1024),
        ]
        self.decoder = [
            VesselDecoder(1024,512),
            VesselDecoder(512,256),
            VesselDecoder(256,128),
            VesselDecoder(128,64),
        ]
        self.last_conv = nn.Conv2d(64,in_channels, kernel_size=(1,1), padding="valid")
        
    def forward(self, x):
        residual_connections = []
        x = self.first_conv(x)
        for encoder in self.encoder:
            x = encoder(x)
            residual_connections.append(x)
        for decoder in self.decoder:
            old_connection = residual_connections.pop()
            x = torch.concat([x, old_connection], dim=1)
            x = decoder(x)
        x = self.last_conv(x)
        return x