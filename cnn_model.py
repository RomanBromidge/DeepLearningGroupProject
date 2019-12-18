import torch
from torch import nn
from torch.nn import functional as F
from typing import NamedTuple


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float,
                    mode: str):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.dropout = nn.Dropout(p=dropout)
        self.dropout2d = nn.Dropout2d(p=dropout)

        #First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            # stride=(2, 2),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv1)
        self.conv1_bn = nn.BatchNorm2d(32)

        #Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            # stride=(2, 2),
            padding=(1, 1)
        )
        self.initialise_layer(self.conv2)
        self.conv2_bn = nn.BatchNorm2d(32)

        #Max pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1,1))

        #Third convolutional layer
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            # stride=(2, 2),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv1)
        self.conv3_bn = nn.BatchNorm2d(64)

        #Fourth convolutional layer
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            # stride=(2, 2),
            padding=(1, 1)
        )
        self.initialise_layer(self.conv2)
        self.conv4_bn = nn.BatchNorm2d(64)

        #Max pooling
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1,1))

        #First fully connected layer
        size = 15488
        if mode = 'MLMC':
            size = 26048
        self.fc1 = nn.Linear(size, 1024)
        self.initialise_layer(self.fc1)
        self.fc1_bn = nn.BatchNorm1d(1024)

        #Final fully connected layer
        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        #Perform the forward pass through the network
        x = F.relu(self.conv1_bn(self.conv1(images)))
        x = self.dropout2d(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool1(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.dropout2d(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(torch.sigmoid(self.fc1_bn(self.fc1(x))))
        x = self.fc2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
