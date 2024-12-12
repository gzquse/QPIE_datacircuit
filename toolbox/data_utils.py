"""
  test data loaders
"""
import re
import time
import os, sys
import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision import datasets


def get_data_loader(params, location, distributed, train=True):
    dataset = TestDataSet(params, location)
    # define a sampler for distributed training using DDP
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    if train:
        batch_size = params.local_batch_size
    else:
        batch_size = params.local_valid_batch_size
    dataloader = DataLoader(dataset,
                            batch_size=int(batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=(sampler is None),
                            sampler=sampler,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())
    return dataloader, dataset, sampler


class TestDataSet(Dataset):
    def __init__(self, params, location):
        self.params = params
        self.location = location
        self.n_samples = 128

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ''' just return random tensors '''
        X = torch.rand((1,128,128))
        y = torch.rand((1,128,128))
        return X, y


############## hard code

# Classical parameters.

sample_count = 1000  # Total number of images to use.
target_digits = [5, 6]  # Hand written digits to classify.
test_size = 30  # Percentage of dataset to be used for testing.
classification_threshold = 0.5  # Classification boundary used to measure accuracy.
epochs = 1000  # Number of epochs to train for.




def prepare_data(target_digits=target_digits, sample_count=sample_count, test_size=test_size):
    """Load and prepare the MNIST dataset to be used

    Args:
        target_digits (list): digits to perform classification of
        sample_count (int): total number of images to be used
        test_size (float): percentage of sample_count to be used as test set, the remainder is the training set

    Returns:
        dataset in train, test format with targets

    """

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307), (0.3081))])

    dataset = datasets.MNIST("./data",
                             train=True,
                             download=True,
                             transform=transform)

    # Filter out the required labels.
    idx = (dataset.targets == target_digits[0]) | (dataset.targets == target_digits[1])
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]

    # Select a subset based on number of datapoints specified by sample_count.
    subset_indices = torch.randperm(dataset.data.size(0))[:sample_count]

    x = dataset.data[subset_indices].float().unsqueeze(1)
    y = dataset.targets[subset_indices].float()

    # Relabel the targets as a 0 or a 1.
    y = torch.where(y == min(target_digits), 0.0, 1.0)

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size / 100,
                                                        shuffle=True,
                                                        random_state=42)

    return x_train, x_test, y_train, y_test