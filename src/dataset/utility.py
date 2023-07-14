import os
import torch 
import requests
import torchvision

import numpy as np

from torch import nn
from PIL import Image
from collections import OrderedDict
from typing import Union, Dict, Tuple
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

def load_weights(
        path_origin: str = './data',
        path_model: str = './model',
    ) -> Tuple[OrderedDict, OrderedDict]:

    path_file = os.path.join(
        path_model, "model_weights.pth"
    )
    path_file_original = os.path.join(
        path_origin, "weights_resnet18_cifar10.pth"
    )  
    weights_unlearn = torch.load(path_file)
    weights_original = torch.load(path_file_original)
    
    return weights_original, weights_unlearn

def load_all_model(
        original_weight: OrderedDict, unlearn_weight: OrderedDict
    ) -> Dict[str, nn.Module]:

    model_original = resnet18(weights=None, num_classes=10)
    model_original.load_state_dict(original_weight)

    model_unlearn = resnet18(weights=None, num_classes=10)
    model_unlearn.load_state_dict(unlearn_weight)
    
    return {
        'original': model_original, 
        'unlearn': model_unlearn
    }