import torch
from torch import nn
import torch.nn.functional as F


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Reduces dimensions by half

        # Fully Connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # Based on input size after convolutions
        self.fc2 = nn.Linear(256, 200)  # Output layer with 200 classes

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, start_dim=1)  # Converts to (batch_size, 128*28*28)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)  # No activation for classification output
        # self.dropout = nn.Dropout(p=0.5)

        return x
