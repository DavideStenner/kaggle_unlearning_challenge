import torch

import numpy as np

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple
from sklearn import linear_model, model_selection

def compute_losses(model: nn.Module, loader: DataLoader, config: dict):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    model = model.to(config['DEVICE'])

    for inputs, targets in loader:

        inputs = inputs.to(config['DEVICE'])
        targets = targets.type(torch.long).to(config['DEVICE'])

        logits = model(inputs)
        losses = criterion(logits, targets).numpy(force=True)

        for l in losses:
            all_losses.append(l)

    model = model.cpu()

    return np.array(all_losses)

def simple_mia(sample_loss: np.ndarray, members: np.ndarray, random_state: int, n_splits: int=10) -> np.ndarray:
    """Computes cross-validation score of a membership inference attack.

    Args:
    sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
    members : array_like of shape (n,),
        whether a sample was used for training.
    n_splits: int
        number of splits to use in the cross-validation.
    Returns:
    scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def create_mia_dataset(forget_losses: np.ndarray, test_losses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    min_dim = min(forget_losses.shape[0], test_losses.shape[0])

    np.random.shuffle(forget_losses)
    np.random.shuffle(test_losses)

    forget_losses, test_losses = forget_losses[:min_dim], test_losses[:min_dim]

    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)
    return samples_mia, labels_mia