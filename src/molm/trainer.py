##TODO: dotenv instead of CAPS_PARAMS !
import logging
import os
import time
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import MSELoss
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision

from .autograd_hacks.autograd_hacks import clear_backprops
from .scores import InceptionScore

SAVING_FREQUENCY = 5

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.Logger("trainer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


class Trainer:
    """Trainer for naive MoLM model"""

    def __init__(
        self,
        generator,
        moment_network,
        train_set,
        training_params,
        device=None,
        scores=None,
        tensorboard=False,
        save_folder="runs/run",
        eval_generate_images=False,
    ):
        """
            generator: a nn.Module child class serving as a generator network
            moment_network: a nn.Module child class serving as the moment network
            loader: a training data loader
            scores: None, or a dict of shape {'name':obj} with score object
                    with a __call__ function that returns a score

            training_params: dict of training parameters with:
                n0: number of objectives
                nm: number of moments trainig step
                ng: number of generating training steps
                lr: learning rate
                beta1 / beta2: Adam parameters
                acw: activation wieghts
                alpha: the norm penalty parameter
                gen_batch_size: the batch size to train the generator
                mom_batch_size: the batch size to train the moment network
                eval_batch_size: the batch size to evaluate the generated
                eval_size: total number of generated samples on which to evaluate the scores

            tensorboard: whether to use tensorboard to save training information
            save_folder: root folder to save the training information
            eval_generate_images: generates images during training for evaluation

        """
        self.G = generator
        self.MoNet = moment_network
        self.train_set = train_set
        self.training_params = training_params
        self.nm = training_params["nm"]
        self.ng = training_params["ng"]
        self.no = training_params["no"]
        self.no_obj = 0  # current objective
        self.n_moments = training_params["n_moments"]
        self.gen_batch_size = training_params["gen_batch_size"]
        self.eval_batch_size = training_params["eval_batch_size"]
        self.learn_moments = training_params["learn_moments"]

        lr, beta1, beta2 = (
            self.training_params["lr"],
            self.training_params["beta1"],
            self.training_params["beta2"],
        )
        self.optimizerG = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizerM = optim.Adam(
            self.MoNet.parameters(), lr=lr, betas=(beta1, beta2)
        )

        self.LM = []
        self.LG = []
        self.iter = 0
        self.device = device

        self.cross_entropy = F.binary_cross_entropy
        self.mse = MSELoss(reduction="sum")

        # to track the evolution of generated samples from a single batch of noises
        self.fixed_z = torch.randn(20, self.G.dims[0], device=self.device)

        # saving training info
        self.run_folder = Path(save_folder)
        if not (self.run_folder / "results").exists():
            os.mkdir(self.run_folder / "results")
        self.save_path_img = self.run_folder / "results/images/"
        self.save_path_checkpoints = self.run_folder / "checkpoints/"
        if not self.save_path_checkpoints.exists():
            os.mkdir(self.save_path_checkpoints)
        self.eval_generate_images = eval_generate_images

        # monitoring the progress of the training with the evaluation scores
        self.scores = scores
        if scores is not None and not (self.run_folder / "scores.csv").exists():
            # Save scores
            with open(self.run_folder / "scores.csv", "w") as f:
                f.write(f'Objective,{",".join(scores.keys())}\n')

        # monitoring through tensorboard
        if tensorboard:
            comment = "".join(
                ["{}={} ".format(key, training_params[key]) for key in training_params]
            )
            self.tb = SummaryWriter(self.run_folder, comment=comment)
            self.tb.add_graph(generator, self.fixed_z)
        else:
            self.tb = None

        # set up handler to file
        fh = logging.FileHandler(self.run_folder / "logging.txt")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def train_monet(self):
        """Solves one Moment Network objective."""
        # reshuffle training data
        loader = iter(
            torch.utils.data.DataLoader(
                self.train_set,
                shuffle=True,
                batch_size=self.training_params["mom_batch_size"],
            )
        )
        for i in range(self.nm):
            batch = loader.next()
            samples = batch
            samples = samples.to(self.device)
            # samples = (samples * 2) - 1

            sample_size = samples.size(0)
            one_labels = torch.ones(sample_size, device=self.device)
            zero_labels = torch.zeros(sample_size, device=self.device)

            # generating latent vector
            # self.dims = [Z_dim, h1_dim, h2_dim, h3_dim, X_dim]
            z = torch.randn(sample_size, self.G.dims[0], device=self.device)
            res = self.G(z)
            prob_trues = self.MoNet(samples)
            output_trues = self.MoNet.output
            prob_gen = self.MoNet(res)
            output_gen = self.MoNet.output

            prob_trues, prob_gen = prob_trues.squeeze(), prob_gen.squeeze()
            LM_samples = self.cross_entropy(prob_trues, one_labels)
            LM_gen = self.cross_entropy(prob_gen, zero_labels)
            LM = LM_samples + LM_gen

            # We now need to compute the gradients to add the regularization term
            mean_output = output_trues.mean()
            self.optimizerM.zero_grad()
            grad_monet = self.MoNet.get_gradients(mean_output)
            grad_monet = grad_monet.squeeze()
            grad_norm = torch.dot(grad_monet, grad_monet)
            LM = (
                LM_samples
                + LM_gen
                + self.training_params["alpha"] * ((grad_norm - 1) ** 2)
            )
            # LM = LM_samples + LM_gen
            # Add to tensorboard
            if self.tb:
                self.tb.add_scalar(
                    "LossMonet/objective_{}".format(self.no_obj + 1), float(LM), i + 1
                )
            self.LM.append(float(LM))
            if i % 50 == 0:
                logger.info(
                    "Moment Network Iteration {}/{}: LM: {:.6}".format(
                        i + 1, self.nm, LM.item()
                    )
                )

            self.optimizerM.zero_grad()
            LM.backward()
            self.optimizerM.step()

            del grad_monet
            del batch

    def eval_true_moments(self):
        """Returns the value of moment vector on observed data."""
        loader = torch.utils.data.DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.training_params["mom_batch_size"],
        )
        # Calculate the moment vector over the entire dataset:
        moments = torch.zeros(self.n_moments, device=self.device)
        for i, batch in enumerate(loader):
            samples = batch
            samples = samples.to(self.device)
            sample_size = samples.size(0)
            # NOT Scaling true images to tanh activation interval:
            # samples = (samples * 2) - 1
            self.optimizerM.zero_grad()
            moments_b = self.MoNet.get_moment_vector(
                samples,
                sample_size,
                weights=self.training_params["activation_weight"],
                detach=True,
            )
            moments = ((i) * moments + moments_b) / (i + 1)
            del batch
            del samples
            del moments_b
        return moments

    def train_generator(self, true_moments):
        """Solves one generator objective."""
        for i in range(self.ng):

            z = torch.randn(self.gen_batch_size, self.G.dims[0], device=self.device)
            res = self.G(z)
            self.optimizerM.zero_grad()
            moments_gz = self.MoNet.get_moment_vector(
                res,
                self.gen_batch_size,
                weights=self.training_params["activation_weight"],
            )
            # moments_gz = ((i) * moments_gz + moments_z) / (i+1)

            del z
            del res

            LG = torch.dot(
                true_moments - moments_gz, true_moments - moments_gz
            )  # equivalent to dot product of difference
            # LG = self.mse(true_moments, moments_gz)
            # Add to tensorboard
            if self.tb:
                self.tb.add_scalar(
                    "LossGenerator/objective_{}".format(self.no_obj + 1),
                    float(LG),
                    i + 1,
                )
            self.LG.append(float(LG))
            if i % 100 == 0:
                logger.info(
                    "Generator Iteration {}/{}: LG: {:.6}".format(
                        i + 1, self.ng, LG.item()
                    )
                )
            self.optimizerG.zero_grad()
            LG.backward()
            self.optimizerG.step()

            del moments_gz

    def generate_and_display(self, z, save=False, save_path=None):
        """"Generates rows of images from latent variable z."""
        # Visualizing the generated images
        examples = self.G(z).detach().cpu()
        examples = examples.reshape(-1, 3, self.G.dims[-1], self.G.dims[-1])
        examples = (examples + 1) / 2
        grid = torchvision.utils.make_grid(examples, nrow=10)  # 10 images per row
        # Add to tensorboard
        if self.tb:
            self.tb.add_image("generated images", grid, self.no_obj)
        fig = plt.figure(figsize=(15, 15))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        if save:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)

    def eval(self):
        """Evaluate generated batch with scores in self.scores"""
        logger.info(f"Evaluating generated samples with scores: {self.scores.keys()}")
        scores_dict = self.scores
        n_loops = self.training_params["eval_size"] // self.eval_batch_size
        results = dict(zip(scores_dict.keys(), [None] * len(scores_dict)))
        for score in scores_dict:
            results[score] = np.zeros(n_loops)
        for i in range(n_loops):
            with torch.no_grad():
                z = torch.randn(
                    self.eval_batch_size, self.G.dims[0], device=self.device
                )
                samples = self.G(z).cpu()
            if "IS" in scores_dict or "FID" in scores_dict:
                samples = InceptionScore.preprocess(samples)
            for score in scores_dict:
                value = scores_dict[score](samples)
                results[score][i] = value if value is not None else np.nan
        for score in scores_dict:
            results[score] = np.nanmean(results[score])
        return results

    def load_from_checkpoints(self, path):
        """
        Loads network parameters and training info from checkpoint
            path: path to checkpoint
        """
        logger.info("Loading network parameters and training info from checkpoint...")
        checkpoint = torch.load(path)
        self.G.load_state_dict(checkpoint["generator_state_dict"])
        self.optimizerG.load_state_dict(checkpoint["optimizerG_state_dict"])
        self.G.train()

        if self.learn_moments:
            self.MoNet.load_state_dict(checkpoint["monet_state_dict"])
            self.optimizerM.load_state_dict(checkpoint["optimizerM_state_dict"])
            self.MoNet.train()

        last_objective = checkpoint["objective"]
        lossG = checkpoint["last_lossG"]
        lossM = checkpoint["last_lossM"]

        return last_objective, lossG, lossM

    def train(self, save_images=False, from_checkpoint=None):
        """Trains naive MoLM model made of generator self.G and
        moment network self.MoNet"""
        if save_images and not self.save_path_img.exists():
            os.mkdir(self.save_path_img)
        last_objective = 0
        if not self.learn_moments:
            true_moments = self.eval_true_moments()

        if from_checkpoint:
            last_objective, lossG, lossM = self.load_from_checkpoints(from_checkpoint)
            logger.info(
                "Starting training from Objective: {}, lossG: {}, lossM: {}".format(
                    last_objective, lossG, lossM
                )
            )

        for i in range(last_objective, self.no):
            # Track the no of objectives solved
            self.no_obj = i

            start = time.time()
            if self.learn_moments:
                logger.info("Training Moment Network...")
                self.train_monet()
                logger.info("Evaluating true moments value...")
                true_moments = self.eval_true_moments()
            logger.info("Training Generator")
            self.train_generator(true_moments)
            self.iter += 1
            stop = time.time()
            duration = (stop - start) / 60

            if self.learn_moments:
                logger.info(
                    "Objective {}/{} - {:.2} minutes: LossMonet: {:.6} LossG: {:.6}".format(
                        i + 1, self.no, duration, self.LM[-1], self.LG[-1]
                    )
                )
            else:
                logger.info(
                    "Objective {}/{} - {:.2} minutes: LossG: {:.6}".format(
                        i + 1, self.no, duration, self.LG[-1]
                    )
                )

            if self.eval_generate_images:
                self.generate_and_display(
                    self.fixed_z,
                    save=save_images,
                    save_path=self.save_path_img
                    + "generated_molm_iter{}.png".format(i),
                )

            if i % SAVING_FREQUENCY == 0:
                logger.info("Saving model ...")
                save_path_checkpoints = self.save_path_checkpoints / f"molm_iter{i}.pt"
                save_dict = {
                    "monet_state_dict": self.MoNet.state_dict(),
                    "generator_state_dict": self.G.state_dict(),
                    "optimizerG_state_dict": self.optimizerG.state_dict(),
                    "objective": i + 1,
                    "last_lossG": self.LG[-1],
                }
                if self.learn_moments:
                    save_dict["last_lossM"] = self.LM[-1]
                    save_dict["optimizerM_state_dict"] = self.optimizerM.state_dict()

                torch.save(save_dict, save_path_checkpoints)

                if self.scores:
                    scores = self.eval()
                    logger.info(f"{scores}")
                    # Add to tensorboard
                    if self.tb:
                        for score in scores:
                            self.tb.add_scalar(
                                "Scores/{}".format(score), scores[score], i + 1
                            )
                    # Save scores
                    with open(self.run_folder / "scores.csv", "a") as f:
                        f.write(f'{i+1},{",".join([str(metric) for metric in scores.values()])}\n')

            # Updating data on tensorboard
            if self.tb:
                for name, param in self.G.named_parameters():
                    self.tb.add_histogram("generator.{}".format(name), param, i + 1)
                    self.tb.add_histogram(
                        "generator.{}.grad".format(name), param.grad, i + 1
                    )
                for name, param in self.MoNet.named_parameters():
                    self.tb.add_histogram("momentNetwork.{}".format(name), param, i + 1)
                    self.tb.add_histogram(
                        "momentNetwork.{}.grad".format(name), param.grad, i + 1
                    )


class TrainerDiagonalW(Trainer):
    """Trainer for MoLM model with diagonal weighting matrix generator loss."""

    def __init__(
        self,
        generator,
        moment_network,
        train_set,
        training_params,
        device=None,
        scores=None,
        tensorboard=False,
        save_folder="runs/run",
        eval_generate_images=False,
    ):
        """
            generator: a nn.Module child class serving as a generator network
            moment_network: a nn.Module child class serving as the moment network
            loader: a training data loader
            scores: None, or a dict of shape {'name':obj}
                    with score object with a __call__ function that returns a score

            training_params: dict of training parameters with:
                n0: number of objectives
                nm: number of moments trainig step
                ng: number of generating training steps
                lr: learning rate
                beta1 / beta2: Adam parameters
                acw: activation wieghts
                alpha: the norm penalty parameter
                gen_batch_size: the batch size to train the generator
                mom_batch_size: the batch size to train the moment network
                eval_batch_size: the batch size to evaluate the generated
                eval_size: total number of generated samples on which to evaluate the scores

            tensorboard: whether to use tensorboard to save training information
            save_folder: root folder to save the training information

        """
        super().__init__(
            generator,
            moment_network,
            train_set,
            training_params,
            device,
            scores,
            tensorboard,
            save_folder,
            eval_generate_images,
        )

    def train_monet(self):
        """Solves one moment network objective."""
        # reshuffle training data
        loader = iter(
            torch.utils.data.DataLoader(
                self.train_set,
                shuffle=True,
                batch_size=self.training_params["mom_batch_size"],
            )
        )
        for i in range(self.nm):
            batch = loader.next()
            samples = batch
            samples = samples.to(self.device)
            # samples = (samples * 2) - 1

            sample_size = samples.size(0)
            one_labels = torch.ones(sample_size, device=self.device)
            zero_labels = torch.zeros(sample_size, device=self.device)

            # generating latent vector
            # self.dims = [Z_dim, h1_dim, h2_dim, h3_dim, X_dim]
            z = torch.randn(sample_size, self.G.dims[0], device=self.device)
            res = self.G(z)
            prob_trues = self.MoNet(samples)
            output_trues = self.MoNet.output
            prob_gen = self.MoNet(res)
            output_gen = self.MoNet.output

            prob_trues, prob_gen = prob_trues.squeeze(), prob_gen.squeeze()
            LM_samples = self.cross_entropy(prob_trues, one_labels)
            LM_gen = self.cross_entropy(prob_gen, zero_labels)
            LM = LM_samples + LM_gen

            # We now need to compute the gradients to add the regularization term
            mean_output = output_trues.mean()
            self.optimizerM.zero_grad()
            clear_backprops(self.MoNet)
            grad_monet = self.MoNet.get_gradients(mean_output)
            grad_monet = grad_monet.squeeze()
            grad_norm = torch.dot(grad_monet, grad_monet)
            LM = (
                LM_samples
                + LM_gen
                + self.training_params["alpha"] * ((grad_norm - 1) ** 2)
            )
            # Add to tensorboard
            if self.tb:
                self.tb.add_scalar(
                    "LossMonet/objective_{}".format(self.no_obj + 1), float(LM), i + 1
                )
            self.LM.append(float(LM))
            if i % 50 == 0:
                logger.info(
                    "Moment Network Iteration {}/{}: LM: {:.6}".format(
                        i + 1, self.nm, LM.item()
                    )
                )

            self.optimizerM.zero_grad()
            clear_backprops(self.MoNet)
            LM.backward()
            self.optimizerM.step()

            del grad_monet
            del batch

    def train_generator(self, true_moments, weights):
        """Solves one generator objective."""
        for i in range(self.ng):
            z = torch.randn(self.gen_batch_size, self.G.dims[0], device=self.device)
            res = self.G(z)
            self.optimizerM.zero_grad()
            clear_backprops(self.MoNet)
            moments_gz = self.MoNet.get_moment_vector(
                res,
                self.gen_batch_size,
                weights=self.training_params["activation_weight"],
            )

            del z
            del res

            # multiplication by a diagonal matrix is equivalent to adding one weight per moment
            diff = true_moments - moments_gz
            LG = torch.sum((diff ** 2) * weights)

            # Add to tensorboard
            if self.tb:
                self.tb.add_scalar(
                    "LossGenerator/objective_{}".format(self.no_obj + 1),
                    float(LG),
                    i + 1,
                )
            self.LG.append(float(LG))
            if i % 100 == 0:
                logger.info(
                    "Generator Iteration {}/{}: LG: {:.6}".format(
                        i + 1, self.ng, LG.item()
                    )
                )
            self.optimizerG.zero_grad()
            LG.backward()
            self.optimizerG.step()

            del moments_gz

    def get_weights(self, true_moments, epsilon=1e-3):
        """Retrieves correlation weights to compute generator learning objective."""
        z = torch.randn(
            self.training_params["eval_batch_size"], self.G.dims[0], device=self.device
        )
        res = self.G(z)
        self.optimizerM.zero_grad()
        moments = self.MoNet.get_moment_matrix(res, detach=True)
        diff = moments - true_moments
        weights = 1 / (torch.var(diff, 0) + epsilon)
        return weights

    def train(self, save_images=False, from_checkpoint=None):
        """Training the model defined by Generator self.G and moment network self.MoNet"""
        last_objective = 0
        if not self.learn_moments:
            true_moments = self.eval_true_moments()

        if from_checkpoint:
            last_objective, lossG, lossM = self.load_from_checkpoints(from_checkpoint)
            logger.info(
                "Starting training from Objective: {}, lossG: {}, lossM: {}".format(
                    last_objective, lossG, lossM
                )
            )

        for i in range(last_objective, self.no):
            # Track the no of objectives solved
            self.no_obj = i

            start = time.time()
            if self.learn_moments:
                logger.info("Training Moment Network...")
                self.train_monet()
                logger.info("Evaluating true moments value...")
                true_moments = self.eval_true_moments()
            logger.info("Training Generator")
            weights = self.get_weights(true_moments)
            self.train_generator(true_moments, weights)
            self.iter += 1
            stop = time.time()
            duration = (stop - start) / 60

            if self.learn_moments:
                logger.info(
                    "Objective {}/{} - {:.2} minutes: LossMonet: {:.6} LossG: {:.6}".format(
                        i + 1, self.no, duration, self.LM[-1], self.LG[-1]
                    )
                )
            else:
                logger.info(
                    "Objective {}/{} - {:.2} minutes: LossG: {:.6}".format(
                        i + 1, self.no, duration, self.LG[-1]
                    )
                )

            if self.eval_generate_images:
                self.generate_and_display(
                    self.fixed_z,
                    save=save_images,
                    save_path=self.save_path_img
                    + "generated_molm_iter{}.png".format(i),
                )

            if i % SAVING_FREQUENCY == 0:
                logger.info("Saving model ...")
                save_path_checkpoints = self.save_path_checkpoints / f"molm_iter{i}.pt"
                save_dict = {
                    "monet_state_dict": self.MoNet.state_dict(),
                    "generator_state_dict": self.G.state_dict(),
                    "optimizerG_state_dict": self.optimizerG.state_dict(),
                    "objective": i + 1,
                    "last_lossG": self.LG[-1],
                }
                if self.learn_moments:
                    save_dict["last_lossM"] = self.LM[-1]
                    save_dict["optimizerM_state_dict"] = self.optimizerM.state_dict()

                torch.save(save_dict, save_path_checkpoints)

                if self.scores:
                    scores = self.eval()
                    logger.info(scores)
                    # Add to tensorboard
                    if self.tb:
                        for score in scores:
                            self.tb.add_scalar(
                                "Scores/{}".format(score), scores[score], i + 1
                            )
                    # Save scores
                    with open(self.run_folder / "scores.csv", "a") as f:
                        f.write(f'{i+1},{",".join([str(metric) for metric in scores.values()])}\n')

            # Updating data on tensorboard
            if self.tb:
                for name, param in self.G.named_parameters():
                    self.tb.add_histogram("generator.{}".format(name), param, i + 1)
                    self.tb.add_histogram(
                        "generator.{}.grad".format(name), param.grad, i + 1
                    )
                for name, param in self.MoNet.named_parameters():
                    self.tb.add_histogram("momentNetwork.{}".format(name), param, i + 1)
                    self.tb.add_histogram(
                        "momentNetwork.{}.grad".format(name), param.grad, i + 1
                    )
