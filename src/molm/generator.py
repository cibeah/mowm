import torch
import torch.nn as nn
import torch.nn.functional as F

from .initializations import init_layer


class Generator(nn.Module):
    """Generator neural network module with convolutional layers."""

    def __init__(
        self, Z_dim=128, X_dim=32, h1_dim=512, h2_dim=256, h3_dim=128, n_channels=3
    ):
        super().__init__()

        self.dims = [Z_dim, h1_dim, h2_dim, h3_dim, X_dim]

        # Fully Connected Linear Layers, no bias for an equivalent to a matmul
        self.fc1 = nn.Linear(in_features=Z_dim, out_features=h1_dim * 4 * 4, bias=False)
        init_layer(self.fc1)

        # Transposed Convolutions
        self.convt1 = nn.ConvTranspose2d(
            h1_dim, h2_dim, kernel_size=4, stride=2, padding=1
        )
        self.convt2 = nn.ConvTranspose2d(
            h2_dim, h3_dim, kernel_size=4, stride=2, padding=1
        )
        self.convt3 = nn.ConvTranspose2d(
            h3_dim, n_channels, kernel_size=4, stride=2, padding=3
        )
        init_layer(self.convt1)
        init_layer(self.convt2)
        init_layer(self.convt3)

        # Batch Normalizations
        self.bnorm1 = nn.BatchNorm2d(h1_dim)
        self.bnorm2 = nn.BatchNorm2d(h2_dim)
        self.bnorm3 = nn.BatchNorm2d(h3_dim)
        init_layer(self.bnorm1, True)
        init_layer(self.bnorm2, True)
        init_layer(self.bnorm3, True)

    def forward(self, t):
        # (1) input layer
        # (2) Latent space vector => project and reshape
        t = self.fc1(t)
        t = t.reshape(-1, self.dims[1], 4, 4)
        t = F.relu(t)
        t = self.bnorm1(t)

        # (3) First fractionnally-strided conv layer
        t = self.convt1(t)
        t = F.relu(t)
        t = self.bnorm2(t)

        # (4) Second fractionnally-strided conv layer
        t = self.convt2(t)
        t = F.relu(t)
        t = self.bnorm3(t)

        # (5) Last transposed conv layer => image generation
        t = self.convt3(t)
        t = torch.tanh(t)

        return t


class FullyConnectedGenerator(nn.Module):
    """Generator neural network module with fully connected linear layers."""

    def __init__(self, Z_dim=128, X_dim=32, h1_dim=512, h2_dim=256, h3_dim=128):
        super().__init__()

        self.dims = [Z_dim, h1_dim, h2_dim, X_dim]

        # Fully Connected Linear Layers, no bias for an equivalent to a matmul
        self.fc1 = nn.Linear(in_features=Z_dim, out_features=h1_dim, bias=False)
        self.fc2 = nn.Linear(in_features=h1_dim, out_features=h2_dim, bias=False)
        self.fc3 = nn.Linear(in_features=h2_dim, out_features=h3_dim, bias=False)
        self.fc4 = nn.Linear(in_features=h3_dim, out_features=X_dim, bias=False)

        # Batch Normalizations
        self.bnorm1 = nn.BatchNorm1d(h1_dim)
        self.bnorm2 = nn.BatchNorm1d(h2_dim)
        self.bnorm3 = nn.BatchNorm1d(h3_dim)

        self.init_weights()

    def init_weights(self):
        """Initializes weights with the
            normal distribution and biases with 0.
        """
        for child in self.children():
            if child.bias is None:
                init_layer(child)
            else:
                init_layer(child, True)

    def forward(self, t):
        # (1) input layer
        # (2) Latent space vector => hidden dimension
        t = self.fc1(t)
        t = F.relu(t)
        t = self.bnorm1(t)

        # (3) Second hidden dimension
        t = self.fc2(t)
        t = F.relu(t)
        t = self.bnorm2(t)

        # (3) Second hidden dimension
        t = self.fc3(t)
        t = F.relu(t)
        t = self.bnorm3(t)

        # (5) Last transposed conv layer => sample generation
        t = self.fc4(t)

        return t
