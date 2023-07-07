#%%
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())

#%%
# download and pre-process CIFAR10
normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=normalize
)
train_loader = DataLoader(train_set, batch_size=128, shuffle=False, num_workers=2)

# we split held out data into test and validation set
held_out = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=normalize
)
test_set, val_set = torch.utils.data.random_split(held_out, [0.2, 0.8])
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)

#%%
# download pre-trained weights
response = requests.get(
    "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/weights_resnet18_cifar10.pth"
)
open("weights_resnet18_cifar10.pth", "wb").write(response.content)
weights_pretrained = torch.load("weights_resnet18_cifar10.pth", map_location=DEVICE)

# load model with pre-trained weights
model = resnet18(weights=None, num_classes=10)
model.load_state_dict(weights_pretrained)
model.to(DEVICE)
model.eval()
# %%
# a temporary data loader without normalization, just to show the images
tmp_dl = DataLoader(
    torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    ),
    batch_size=16 * 5,
)
images, labels = next(iter(tmp_dl))

fig, ax = plt.subplots(figsize=(12, 6))
plt.title("Sample images from CIFAR10 dataset")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
plt.show()

# %%
def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


print(f"Train set accuracy: {100.0 * accuracy(model, train_loader):0.1f}%%")
print(f"Test set accuracy: {100.0 * accuracy(model, test_loader):0.1f}%%")

# %%

forget_set, retain_set = torch.utils.data.random_split(train_set, [0.1, 0.9])
forget_loader = torch.utils.data.DataLoader(
    forget_set, batch_size=20, shuffle=False, num_workers=2
)
retain_loader = torch.utils.data.DataLoader(
    retain_set, batch_size=180, shuffle=False, num_workers=2
)

#%%

retain_target = []

for inputs, _ in retain_loader:
    inputs = inputs.to(DEVICE)
    retain_target.append(model(inputs).detach())

def unlearning(net, retain_target, retain, forget, validation):
    """Unlearning by fine-tuning.

    Fine-tuning is a very simple algorithm that trains using only
    the retain set.

    Args:
    net : nn.Module.
        pre-trained model to use as base of unlearning.
    retain : torch.utils.data.DataLoader.
        Dataset loader for access to the retain set. This is the subset
        of the training set that we don't want to forget.
    forget : torch.utils.data.DataLoader.
        Dataset loader for access to the forget set. This is the subset
        of the training set that we want to forget. This method doesn't
        make use of the forget set.
    validation : torch.utils.data.DataLoader.
        Dataset loader for access to the validation set. This method doesn't
        make use of the validation set.
    Returns:
    net : updated model
    """
    epochs = 3

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()

    for _ in tqdm(range(epochs)):
        forget_iter = iter(forget)
        for k, (inputs, targets) in enumerate(retain):
            
            inputs = inputs.to(DEVICE)
            inputs_forget, _ = next(forget_iter)
            
            target_retain = retain_target[k].to(DEVICE)
            target_forget = (torch.ones((20, 10))*-2.19722).to(DEVICE)
            
            outputs_retain = net(inputs)
            outputs_forget = net(inputs_forget.to(DEVICE))
            
            outputs = torch.concat((outputs_retain, outputs_forget))
            targets = torch.concat((target_retain, target_forget))

            optimizer.zero_grad()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        scheduler.step()

    net.eval()
    return net

#%%
model_ft = resnet18(weights=None, num_classes=10)
model_ft.load_state_dict(weights_pretrained)
model_ft.to(DEVICE)

# Execute the unlearing routine. This might take a few minutes.
# If run on colab, be sure to be running it on  an instance with GPUs
model_ft = unlearning(model_ft, retain_target, retain_loader, forget_loader, test_loader)

#%%
print(f"Retain set accuracy: {100.0 * accuracy(model_ft, retain_loader):0.1f}%%")
print(f"Test set accuracy: {100.0 * accuracy(model_ft, test_loader):0.1f}%%")

#%%
def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)


train_losses = compute_losses(model, train_loader)
test_losses = compute_losses(model, test_loader)

#%%
# plot losses on train and test set
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
plt.show()

#%%
def simple_mia(sample_loss, members, n_splits=10, random_state=0):
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

#%%
forget_losses = compute_losses(model, forget_loader)

# Since we have more forget losses than test losses, sub-sample them, to have a class-balanced dataset.
np.random.shuffle(forget_losses)
forget_losses = forget_losses[:len(test_losses)]

samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)

mia_scores = simple_mia(samples_mia, labels_mia)

print(
    f"The MIA attack has an accuracy of {mia_scores.mean():.3f} on forgotten vs unseen images"
)

#%%
ft_forget_losses = compute_losses(model_ft, forget_loader)
ft_test_losses = compute_losses(model_ft, test_loader)

# Since we have more forget losses than test losses, sub-sample them, to have a class-balanced dataset.
np.random.shuffle(ft_forget_losses)
ft_forget_losses = ft_forget_losses[:len(ft_test_losses)]

samples_mia_ft = np.concatenate((ft_test_losses, ft_forget_losses)).reshape((-1, 1))
labels_mia = [0] * len(ft_test_losses) + [1] * len(ft_forget_losses)

#%%
mia_scores_ft = simple_mia(samples_mia_ft, labels_mia)

print(
    f"The MIA attack has an accuracy of {mia_scores_ft.mean():.3f} on forgotten vs unseen images"
)

#%%
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
plt.show()