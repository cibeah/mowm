import argparse
import json

import torch

from molm.dataset import load_pickle
from molm.generator import FullyConnectedGenerator
from molm.linear_moment_network import LinearMomentNetwork, PerSampleLinearMomentNetwork
from molm.trainer import Trainer, TrainerDiagonalW
from molm.scores import load_scores
from molm.utils import get_logger, get_n_params

logger = get_logger("Train")

Z_DIM = 64
X_DIM = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "params_path", type=str, help="path to the .json parameters file"
    )
    parser.add_argument(
        "--scores",
        type=str,
        nargs="*",
        default=["FID", "IS"],
        help="scores used to evaluate the model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../data/CIFAR",
        help="path to the training data. Defauls to ../data/CIFAR",
    )
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
    parser.add_argument(
        "--dimensions", type=str, nargs="*", help="Hidden dimensions",
    )

    args = parser.parse_args()
    params_path = args.params_path
    with open(params_path, "r") as f:
        params_dict = json.load(f)
    dataset = args.dataset
    device_name = args.device
    scores = args.scores
    savepath = args.savepath
    dimensions = args.dimensions if args.dimensions else [512, 256, 128]

    logger.info(
        f"\n Launching training run with parameters: \n {json.dumps(params_dict, indent=4)}, \
        \n -- training dataset: {dataset}, \n -- device: {device_name}, \
        \n -- scores: {scores}, \n -- save folder: {savepath}"
    )

    train_set = load_pickle(dataset)
    device = torch.device(device_name)
    scores_dict = load_scores(scores, samples=train_set.T, device=device)

    torch.manual_seed(0)
    G = FullyConnectedGenerator(
        Z_dim=Z_DIM,
        X_dim=X_DIM,
        h1_dim=dimensions[0],
        h2_dim=dimensions[1],
        h3_dim=dimensions[2],
    ).to(device)
    logger.info(f'Number of Parameters in G: {get_n_params(G)}')
    torch.manual_seed(0)
    MoNet = PerSampleLinearMomentNetwork(
        X_dim=X_DIM, h1_dim=dimensions[0], h2_dim=dimensions[0], h3_dim=dimensions[1]
    ).to(device)

    # trainer = Trainer(
    #     G,
    #     MoNet,
    #     train_set,
    #     params_dict,
    #     device,
    #     scores=scores_dict,
    #     tensorboard=False,
    #     save_folder=savepath,
    #     eval_generate_images=False
    # )

    trainer = TrainerDiagonalW(
        G,
        MoNet,
        train_set,
        params_dict,
        device,
        scores=scores_dict,
        tensorboard=False,
        save_folder=savepath,
        eval_generate_images=False,
    )

    if args.checkpoint:
        logger.info(
            "\n Resuming training from checkpoint: \
        \n {}".format(
                params_dict, dataset, device_name, scores
            )
        )
        trainer.train(from_checkpoint=args.checkpoint)
    else:
        trainer.train()
