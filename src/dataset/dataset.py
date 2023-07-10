import torch

import numpy as np

from PIL import Image
from typing import Optional
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self,
        data: np.ndarray, targets: Optional[list] = None
    ):
        self.data = data
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), 
                    (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        if targets is None:
            self.inference = True

        else:
            self.inference=False
            self.targets = targets



    def prepare_input(self, img: np.ndarray) -> torch.tensor:
        img = Image.fromarray(img)
        img = self.transform(img)
        return img
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        img_input = self.prepare_input(self.data[index])

        if self.inference:
            return img_input
        else:
            target = self.targets[index]

            return img_input, target
