import copy

import numpy as np
from scipy import linalg
from scipy.stats import gaussian_kde
import torch
from torch import randperm
from torchvision import transforms

from .utils import get_logger

logger = get_logger("Scores")


class Score:
    """Base Score class to comoute metrics."""

    def __init__(self, device=None):
        if device:
            self.device = device

    def __call__(self, batch):
        score = None
        return score


class Identity(torch.nn.Module):
    """Dummy Identity module implementing (nn(x) = x)"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class InceptionScore(Score):
    """Callable class implementing the Inception Score"""

    transforms_pipe = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    toPILImage = transforms.ToPILImage()

    def __init__(self, model, device=None):
        super().__init__(device)
        self.model = model
        self.device = device
        if device:
            self.model = self.model.to(self.device)

    @classmethod
    def preprocess(cls, batch):
        """Preprocesses batch of images to fit Inception model's input range."""
        batch = (batch + 1) / 2
        ##TODO: Implement transforms directly on Tensor without conversion to PIL Image
        imgs = [
            cls.transforms_pipe(cls.toPILImage(sample)).unsqueeze(0) for sample in batch
        ]
        samples = torch.cat(imgs, dim=0)
        return samples

    def compute_score(self, probs, eps=1e-16):
        """Compute the score from the model's output probabilities."""
        # compute marginal probabilities
        marginal_probs = probs.mean(0)

        # compute KL divergence for each image
        kl = probs * (np.log(probs + eps) - np.log(marginal_probs + eps))
        kl = kl.sum(1)

        # we average the KL divergences over the images
        kl_mean = kl.mean(0)
        score = np.exp(kl_mean)
        return score

    def __call__(self, batch):
        """
        :returns: the inception score for the batch of images
        """
        ##TODO: Check that it's a batch and throw error or unsqueeze
        ## batch = batch.unsqueeze(0)
        # batch = self.preprocess(batch)
        if self.device:
            batch = batch.to(self.device)
        with torch.no_grad():
            output = self.model(batch)
        probs = output.cpu()
        # probs = torch.nn.functional.softmax(output, dim=1).cpu()
        score = self.compute_score(probs)
        return float(score)


class FID(Score):
    """Callable class implementing the Fréchet Inception Distance"""

    def __init__(self, model, device=None):
        super().__init__(device)
        self.model = model
        self.device = device
        self.fitted = False
        self.reference_mu = np.zeros((1, 2048))
        self.reference_sigma = np.zeros((2048, 2048))
        if device:
            self.model = self.model.to(device)

    @staticmethod
    def make_fid_model(model):
        """Returns the cropped version of the Inception model used to compute FID."""
        model_fid = copy.deepcopy(model)
        # replace last fc layer with identity layer
        model_fid.fc = Identity()
        model_fid.eval()
        return model_fid

    def fit(self, data, r=0.2):
        """
        Compute real data features
        data: iterable data loader
        """
        # TODO: maybe use an argument r as the % of real data needed to compute real statistics
        for n, batch in enumerate(data):
            samples, _ = batch
            if self.device:
                samples = samples.to(self.device)
            with torch.no_grad():
                real_output = self.model(samples)
            if self.device:
                real_output = real_output.cpu().numpy()
            self.reference_mu += np.mean(real_output, 0)
            self.reference_sigma += np.cov(real_output, rowvar=False)
        # TODO: actual incremental calculation of sigma
        # (mu is ok since all batches have the same size except 1)
        self.reference_mu /= len(data)
        self.reference_sigma /= len(data)
        self.fitted = True

    def compute_score(self, ouput, eps=1e-16):
        """
        Compute FID on generated data.
        formula:  ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2))
        cf. https://github.com/bioinf-jku/TTUR/blob/master/fid.py
        Vectors must be on cpu.
        """
        # compute marginal probabilities
        if not self.fitted:
            raise TypeError(
                "No reference statistics. Fit on real data before computing the score."
            )
        mu = np.mean(ouput, 0)
        sigma = np.cov(ouput, rowvar=False)
        covmean, _ = linalg.sqrtm(self.reference_sigma.dot(sigma), disp=False)

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        mu_diff = self.reference_mu - mu
        score = mu_diff.dot(mu_diff.T) + np.trace(
            self.reference_sigma + sigma - 2 * covmean
        )
        return score

    def __call__(self, batch):
        """:returns: the Fréchet Inception Distance computed on the generated batch."""
        if self.device:
            batch = batch.to(self.device)
        with torch.no_grad():
            output = self.model(batch)
        output = output.cpu().numpy()
        score = self.compute_score(output)
        return float(score)


class KL(Score):
    """Callable class implementing the empirical KL and JS divergences based
    on two batches of samples"""

    def __init__(self, reference=None):
        """
        :param reference: reference data from which to compute KL distance. Shape (data, 2).
        """
        super().__init__()
        self.reference = None
        if reference is not None:
            self.reference = reference
            self.ref_kde = self._get_kde(reference.T)

    def _get_kde(self, data):
        """
        :param data: batch of samples
        :returns: a callable function computed from :param data: that returns
        an estimate of a density at a given position """
        try:
            return gaussian_kde(data)
        except np.linalg.linalg.LinAlgError as e:
            logger.warning(f"Could not compute KDE: {e}")
            return None

    def _compute_kl(self, p, p_kde, q_kde, bins=100):
        """
        :p: batch of samples from the p distribution, shape(2, n_samples)
        :p_kde: callable function estimating density of p
        :q_kde: callable function estimating density of q
        :bins: number of points at which to evalutate density estimates
        :returns: empirical estimation of the KL divergence between p and q distributions
        """
        xmin, xmax = p[0].min().item(), p[0].max().item()
        ymin, ymax = p[1].min().item(), p[1].max().item()

        deltax = float((xmax - xmin) / bins)
        deltay = float((ymax - ymin) / bins)

        X, Y = np.mgrid[xmin:xmax:deltax, ymin:ymax:deltay]
        positions = np.vstack([X.ravel(), Y.ravel()])

        px = p_kde(positions)
        qx = q_kde(positions)
        f = px * (np.log(px) - np.log(qx)) * deltax * deltay
        return f.sum()

    def __call__(self, p, q=None, bins=100):
        """
        :param p: batch of samples from distribution p, shape (n_samples, 2)
        :param q: batch of samples from reference distribution q.
                  Only needed if the class was not already initialiezed with a reference.
        :param bins: number of points at which to evalutate density estimates
        :returns: empirical estimation of the KL divergence between
                  p and q / reference distributions
        """
        if self.reference is None and not q is None:
            raise ValueError(
                "KL has no reference. Two samples must be passed as arguments. "
            )
        if self.reference is not None and q is not None:
            raise ValueError(
                "Two samples were provided, but KL already has a reference sample."
            )
        if q is not None and q.shape[1] != 2:
            raise ValueError(
                f"Invalid input dimensions. compute_JS takes 2D samples as inputs, but \
                a tensor of shape {q.shape} was passed."
            )

        p_kde = self._get_kde(p.T)
        q_kde = self.ref_kde if self.reference is not None else self._get_kde(q.T)
        if p_kde is None or q_kde is None:
            logger.warning("Could not compute KL divergence on samples.")
            return None
        return self._compute_kl(p.T, p_kde, q_kde, bins=bins)

    def compute_JS(self, p, q=None, bins=100):
        """
        :p: batch of samples from the p distribution
        :q:  batch of samples from the q distribution.
             Only needed if the class was not already initialiezed with a reference.
        :bins: number of points at which to evalutate density estimates
        :returns: empirical estimation of the JS divergence between p and q distributions
        """
        if self.reference is None and q is None:
            raise ValueError(
                "KL has no reference. Two samples must be passed as arguments. "
            )
        if self.reference is not None and q is not None:
            raise ValueError(
                "Two samples were provided, but KL already has a reference sample."
            )

        if q is not None and q.shape[1] != 2:
            raise ValueError(
                f"Invalid input dimensions. compute_JS takes 2D samples as inputs, but \
                a tensor of shape {q.shape} was passed."
            )

        if self.reference is not None:
            q = self.reference
            q_kde = self.ref_kde
        else:
            q_kde = self._get_kde(q.T)

        if q.shape != p.shape:
            logger.debug(
                f"compute_JS was passed two batches of different shapes. \
            Inputs will be downsided to the smaller input."
            )
            if q.shape[0] > p.shape[0]:
                indices = randperm(q.shape[0])[: p.shape[0]]
                q = q[indices]
            if p.shape[0] > q.shape[0]:
                indices = randperm(p.shape[0])[: q.shape[0]]
                p = p[indices]

        m = (p + q) / 2
        p_kde = self._get_kde(p.T)
        m_kde = self._get_kde(m.T)

        if m_kde is None or q_kde is None or p_kde is None:
            logger.warning("Could not compute JS divergence on samples.")
            return None

        JS = (
            self._compute_kl(p.T, p_kde, m_kde, bins=bins) / 2
            + self._compute_kl(q.T, q_kde, m_kde, bins=bins) / 2
        )
        return JS


def load_scores(scores, samples=None, device=None):
    """:returns: dictionary with score names as key,
                 and their callable functions as values"""
    scores_dict = {}
    if "FID" in scores or "IS" in scores:
        inception_v3 = torch.hub.load(
            "pytorch/vision:v0.5.0", "inception_v3", pretrained=True
        )
        inception_v3.eval()

    if "FID" in scores:
        logger.info("FID will be used for scoring")
        model_fid = FID.make_fid_model(inception_v3)
        fid = FID(model_fid)
        logger.info("Fitting FID to observed data.")
        fid.fit(samples)
        scores_dict["FID"] = fid

    elif "IS" in scores:
        logger.info("Inception Score will be used for scoring")
        scores_dict["IS"] = InceptionScore(inception_v3, device)

    elif "KL" in scores or "JS" in scores:
        logger.info("KL ans JS divergences will be used for scoring")
        scoring = KL(samples.T)
        scores_dict["KL"] = scoring
        scores_dict["JS"] = scoring.compute_JS

    else:
        logger.info("No score has been passed for evaluation.")

    return scores_dict
