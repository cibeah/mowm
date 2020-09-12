import torch
import torch.nn as nn

from .initializations import init_layer
from .autograd_hacks.autograd_hacks import (
    add_hooks,
    disable_hooks,
    enable_hooks,
    clear_backprops,
    compute_grad1,
)


class BaseMomentNetwork(nn.Module):
    """Base class to support all implemented versions of the Moment Network"""

    def __init__(self):
        super().__init__()
        # keep track of last hidden layer values
        self.h = []
        self.output = None

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
        raise NotImplementedError(
            "This is an instance of a base Moment Network class, \
            please use an implemented child class."
        )

    def get_gradients(self, fn):
        """
        Returns a concatenated vector of the sums of all gradients wrt to a function fn.
        fn: tensor with a graph
        """
        fn.backward(retain_graph=True)
        grads = []
        for param_tensor in self.parameters():
            grads.append(param_tensor.grad.reshape(1, -1))

        gradients = torch.cat(grads, dim=1)
        return gradients

    def get_moment_vector(self, x, size, weights=1e-4, detach=False):
        """
        Returns a concatenated vector of all moments to be used for training with
        moment matching: based on concatenated activations layers and gradients .
        """
        self.forward(x)
        output, hidden = self.output, self.h

        mean_output = output.mean()
        grad_monet = self.get_gradients(mean_output)
        grad_monet = (grad_monet / size).squeeze()
        if detach:
            hidden = [h.detach() for h in hidden]
            grad_monet = grad_monet.detach()
            x = x.detach()

        activations = torch.cat(hidden, dim=1)
        mean_activations = activations.mean(0) * weights
        moments = torch.cat([grad_monet, mean_activations], dim=0)
        # moments = torch.cat([grad_monet, mean_activations, x.mean(0)], dim=0)
        return moments


class BasePerSampleMomentNetwork(BaseMomentNetwork):
    """Base class to support all implemented versions of the Per Sample Moment Network"""

    def __init__(self):
        super().__init__()
        # add layers
        # call self._set_up_hooks()

    def _set_up_hooks(self):
        """Sets up per-sample gradient tracking. To be called in the
        init function of the child class."""
        # Add per-sample gradient tracking
        add_hooks(self)
        # Only activate the hooks when needed to limit memory usage
        disable_hooks()

    def get_gradient_matrix(self, fn):
        """
        Returns a tensor of shape (batch size, gradients) of the sums of all
        gradients wrt to a function fn.
        :param fn: scalar tensor with a graph of shape
        """
        fn.backward(retain_graph=True)
        compute_grad1(self)
        grads = []
        for param_tensor in self.parameters():
            grads.append(param_tensor.grad1.reshape(self.output.size(0), -1))

        gradients = torch.cat(grads, dim=1)
        return gradients

    def get_moment_matrix(self, x, weights=1e-4, detach=False):
        """
        Returns a tensor of shape (batch size, gradients) of the sums of all
        gradients wrt to a function fn.
        :param fn: scalar tensor with a graph of shape
        """
        enable_hooks()
        self.forward(x)
        output, hidden = self.output, self.h
        mean_output = output.mean()
        grad_monet = self.get_gradient_matrix(mean_output)
        clear_backprops(self)
        disable_hooks()
        if detach:
            hidden = [h.detach() for h in hidden]
            grad_monet = grad_monet.detach()
            x = x.detach()

        moments = torch.cat([grad_monet] + hidden, dim=1)
        # moments = torch.cat([grad_monet, x] + hidden, dim=1)
        return moments


class MomentNetwork(BaseMomentNetwork):
    """Moment Network Class with convolutional layers"""

    def __init__(
        self, Z_dim=128, X_dim=32, h1_dim=512, h2_dim=256, h3_dim=128, n_channels=3
    ):
        super().__init__()

        self.Z_dim = Z_dim
        self.X_dim = X_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.h3_dim = h3_dim

        # Size-Preserving Convolutions
        self.conv11 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=h3_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv21 = nn.Conv2d(
            in_channels=h3_dim, out_channels=h2_dim, kernel_size=3, stride=1, padding=1
        )
        self.conv31 = nn.Conv2d(
            in_channels=h2_dim, out_channels=h1_dim, kernel_size=3, stride=1, padding=1
        )
        init_layer(self.conv11)
        init_layer(self.conv21)
        init_layer(self.conv31)

        # Stride 2 Convolutions
        self.conv12 = nn.Conv2d(
            in_channels=h3_dim, out_channels=h3_dim, kernel_size=3, stride=2, padding=1
        )
        self.conv22 = nn.Conv2d(
            in_channels=h2_dim, out_channels=h2_dim, kernel_size=3, stride=2, padding=1
        )
        self.conv32 = nn.Conv2d(
            in_channels=h1_dim, out_channels=h1_dim, kernel_size=3, stride=2, padding=1
        )
        init_layer(self.conv12)
        init_layer(self.conv22)
        init_layer(self.conv32)

        # Output Linear Layer
        self.fc = nn.Linear(in_features=h1_dim * 4 * 4, out_features=1)

        # Leaky ReLus:
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, t):
        self.h = []
        size = t.size(0)
        # (1) input layer
        # (1) size-preserving + stride-2
        t = self.conv11(t)
        t = self.lrelu(t)
        self.h.append(t.reshape(size, -1))
        # print(t.shape)
        t = self.conv12(t)
        # print(t.shape)
        t = self.lrelu(t)
        self.h.append(t.reshape(size, -1))

        # (2) size-preserving + stride-2
        t = self.conv21(t)
        t = self.lrelu(t)
        self.h.append(t.reshape(size, -1))
        # print(t.shape)
        t = self.conv22(t)
        # print(t.shape)
        t = self.lrelu(t)
        self.h.append(t.reshape(size, -1))

        # (3) size-preserving + stride-2
        t = self.conv31(t)
        t = self.lrelu(t)
        self.h.append(t.reshape(size, -1))
        # print(t.shape)
        t = self.conv32(t)
        # print(t.shape)
        t = self.lrelu(t)
        # self.h.append(t.reshape(size, -1))

        # (4) reshape + Linear Layer + sigmoid activation
        t = t.reshape(size, -1)
        t = self.fc(t)
        self.output = t
        t = torch.sigmoid(t)

        return t


class PerSampleMomentNetwork(MomentNetwork, BasePerSampleMomentNetwork):
    """Moment Network Class with per sample gradients and convolutional layers"""

    def __init__(
        self, Z_dim=128, X_dim=32, h1_dim=512, h2_dim=256, h3_dim=128, n_channels=3
    ):
        super().__init__(
            Z_dim=Z_dim,
            X_dim=X_dim,
            h1_dim=h1_dim,
            h2_dim=h2_dim,
            h3_dim=h3_dim,
            n_channels=n_channels,
        )
        self._set_up_hooks()
