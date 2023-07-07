import os
import torch 
import requests
import torchvision

import numpy as np

from torch import nn
from PIL import Image
from typing import Union, Dict
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split

def load_dataset(
        config: dict, 
        download: bool = False, 
        root: str='./data'
    ) -> Dict[
        str, Dict[str, Union[np.ndarray, list]]
    ]:
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=download, transform=normalize
    )

    # we split held out data into test and validation set
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=download, transform=normalize
    )
    train_data, train_target = train_set.data, train_set.targets
    test_data, test_target = test_set.data, test_set.targets

    (
        retain_data, forget_data, 
        retain_target, forget_target
    ) = train_test_split(train_data, train_target, test_size=0.1, random_state=config["RANDOM_STATE"])

    results = {
        "train": {"data": train_data, "targets": train_target},
        "test": {"data": test_data, "targets": test_target},
        "retain": {"data": retain_data, "targets": retain_target},
        "forget": {"data": forget_data, "targets": forget_target}
    }
    return results

def load_model(
        config: dict,
        root: str = './data',
    ) -> nn.Module:

    path_file = os.path.join(
        root, "weights_resnet18_cifar10.pth"
    )
    #save file
    if not os.path.exists(path_file):
        response = requests.get(config['MODEL_URL'])
        with open(path_file, "wb") as file:
            file.write(response.content)
        
    weights_pretrained = torch.load(path_file)
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    return model
