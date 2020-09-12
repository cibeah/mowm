import torch
import torch.nn as nn
import torch.nn.functional as F

from .moment_network import BaseMomentNetwork, BasePerSampleMomentNetwork


class LinearMomentNetwork(BaseMomentNetwork):
    """Moment Network Class with linear layers"""

    def __init__(self, X_dim, h1_dim, h2_dim, h3_dim):
        super().__init__()
        self.X_dim = X_dim
        self.fc1 = nn.Linear(in_features=X_dim, out_features=h1_dim)
        self.fc2 = nn.Linear(in_features=h1_dim, out_features=h2_dim)
        self.fc3 = nn.Linear(in_features=h2_dim, out_features=h3_dim)
        self.fc4 = nn.Linear(in_features=h3_dim, out_features=1)

        # self.init_weights()

    def forward(self, t):
        self.h = []
        # (1) input layer
        # (2) latent space vector => hidden dimension
        t = t.reshape(-1, self.X_dim)
        t = self.fc1(t)
        self.h.append(t)
        t = F.relu(t)

        # (2) hidden layer to hidden layer
        t = self.fc2(t)
        self.h.append(t)
        t = F.relu(t)

        # (3) output linear layer
        t = self.fc3(t)
        self.h.append(t)
        t = F.relu(t)

        # (3) output linear layer
        t = self.fc4(t)
        self.output = t
        t = torch.sigmoid(t)

        return t


class PerSampleLinearMomentNetwork(LinearMomentNetwork, BasePerSampleMomentNetwork):
    """Moment Network Class with per sample gradients and linear layers"""

    def __init__(self, X_dim, h1_dim, h2_dim, h3_dim):
        super().__init__(X_dim=X_dim, h1_dim=h1_dim, h2_dim=h2_dim, h3_dim=h3_dim)
        # LinearMomentNetwork.__init__(
        # )
        self._set_up_hooks()
