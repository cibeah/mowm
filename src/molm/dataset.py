import argparse
import logging
import pickle

from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

logger = logging.Logger("Datasets")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class RewindIterator(DataLoader):
    """
    Data Loader repeating the dataset up to the given number of iterations
    """

    def __init__(self, dataset, batch_size, iterations=1000, shuffle=True):
        self.iterations = iterations
        super().__init__(dataset, shuffle=shuffle, batch_size=batch_size)
        self.iterator = super().__iter__()

    def __len__(self):
        return self.iterations

    def iterate(self):
        for _ in range(self.iterations):
            try:
                batch = next(self.iterator)
            except StopIteration:
                # iterator exhausted
                iterator = super().__iter__()
                batch = next(iterator)
            yield batch


def load_cifar(dataset):
    return torchvision.datasets.CIFAR10(
        root=dataset,
        train=True,
        download=True,  # download if non existant at the location 'root'
        transform=transforms.Compose(
            [transforms.ToTensor()]  # we want our data to be loaded as tensors
        ),
    )


def load_pickle(dataset):
    with open(dataset, "rb") as f:
        return pickle.loads(f.read())
