import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader

from src.loss.loss import compute_losses

def train_test_loss_plot(train_losses: np.ndarray, test_losses: np.ndarray) -> plt.figure:
    fig = plt.figure()

    plt.title("Losses on train and test set (pre-trained model)")

    plt.hist(test_losses, density=True, alpha=0.5, bins=50, label="Test set")
    plt.hist(train_losses, density=True, alpha=0.5, bins=50, label="Train set")
    
    plt.xlabel("Loss", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    
    plt.xlim((0, np.max(test_losses)))
    plt.yscale("log")
    
    plt.legend(frameon=False, fontsize=14)
    
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return fig

def attack_score_plot(
        config: dict,
        original_model: nn.Module, unlearn_model: nn.Module, 
        test_loader: DataLoader, forget_loader: DataLoader,
        mia_scores: np.ndarray, mia_scores_ft: np.ndarray
    ) -> plt.figure:
    
    ft_forget_losses = compute_losses(model=original_model, loader=forget_loader, config=config)
    ft_test_losses = compute_losses(model=original_model, loader=test_loader, config=config)

    forget_losses = compute_losses(model=unlearn_model, loader=forget_loader, config=config)
    test_losses = compute_losses(model=unlearn_model, loader=test_loader, config=config)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.set_title(f"Pre-trained model.\nAttack accuracy: {mia_scores.mean():0.2f}")
    ax1.hist(test_losses, density=True, alpha=0.5, bins=50, label="Test set")
    ax1.hist(forget_losses, density=True, alpha=0.5, bins=50, label="Forget set")

    ax2.set_title(f"Unlearned model.\nAttack accuracy: {mia_scores_ft.mean():0.2f}")
    ax2.hist(ft_test_losses, density=True, alpha=0.5, bins=50, label="Test set")
    ax2.hist(ft_forget_losses, density=True, alpha=0.5, bins=50, label="Forget set")

    ax1.set_xlabel("Loss")
    ax2.set_xlabel("Loss")
    ax1.set_ylabel("Frequency")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_xlim((0, np.max(test_losses)))
    ax2.set_xlim((0, np.max(test_losses)))

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    ax1.legend(frameon=False, fontsize=14)

    return fig