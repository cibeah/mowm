import argparse

import matplotlib.pyplot as plt
import torch

from molm.generator import FullyConnectedGenerator
from molm.linear_moment_network import LinearMomentNetwork
from molm.utils import get_logger

logger = get_logger("Eval")

Z_DIM = 64
X_DIM = 2
dimensions = [512, 256, 128]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="name of the device to be used for training (only one device cause we aren't rich)",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        help="path to a checkpoint from which the training should resume",
    )
    parser.add_argument(
        "--savepath",
        "-s",
        type=str,
        help="path to a folder to save training checkpoint and tensorboard file",
    )

    args = parser.parse_args()
    checkpoint = args.checkpoint
    device_name = args.device
    savepath = args.savepath
    device = torch.device(device_name)

    G = FullyConnectedGenerator(
        Z_dim=Z_DIM,
        X_dim=X_DIM,
        h1_dim=dimensions[0],
        h2_dim=dimensions[1],
        h3_dim=dimensions[2],
    ).to(device)
    MoNet = LinearMomentNetwork(
        X_dim=X_DIM, h1_dim=dimensions[0], h2_dim=dimensions[0], h3_dim=dimensions[1]
    ).to(device)

    z = torch.randn(5000, G.dims[0], device=device)
    checkp = torch.load(checkpoint)
    G.load_state_dict(checkp["generator_state_dict"])

    samples = G(z).detach().cpu()
    # output = MoNet(samples)
    # # output.mean(0).backward()
    # grads = MoNet.get_gradients(MoNet.output.mean(0))

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.scatter(samples[:, 0], samples[:, 1], s=1)
    ax.set_xlim(-4, 4)
    ax.set_xlim(-6, 6)
    # plt.savefig(savepath)
    plt.show()
