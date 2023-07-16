import torch

import numpy as np

from tqdm import tqdm
from PIL import Image
from typing import Optional, Tuple, Dict
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold

from src.utils import free_memory
from src.dataset.utility import load_model

COEF_K = -2.19722
class ImageDataset(Dataset):
    def __init__(self,
        data: np.ndarray, targets: Optional[list|np.ndarray] = None
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
            target = torch.tensor(self.targets[index], dtype=torch.float)

            return img_input, target

class MonitorImageDataset(ImageDataset):
    def __init__(self,
        data: Dict[str, np.ndarray]
    ):
        super().__init__(data=data['data'], targets=data['targets'])

        self.labels_which_dataset = data['labels_which_dataset']
    
    def __getitem__(self, index):
        img_input = self.prepare_input(self.data[index])

        target = torch.tensor(self.targets[index], dtype=torch.float)
        label_dataset = self.labels_which_dataset[index]

        return img_input, target, label_dataset


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y: np.ndarray, batch_size: int, shuffle: bool):

        self.n_batches = int(len(y) / batch_size)

        self.skf = StratifiedKFold(n_splits=self.n_batches, shuffle=shuffle)
        
        self.y = ((y == COEF_K).mean(axis=1) > 0).astype(int)

        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
            
        for _, test_idx in self.skf.split(self.y, self.y):
            yield test_idx

    def __len__(self):
        return self.n_batches

def get_unlearning_dataset(
        config:dict, 
        dataset: dict, get_all: bool = False, 
        coef_k: float=COEF_K
    ) -> Tuple[np.ndarray, np.ndarray]:

    retain_img_dataset = ImageDataset(dataset['retain']['data'])
    inference_retain = DataLoader(
        retain_img_dataset, 
        batch_size=256, shuffle=False, num_workers=1,
        pin_memory=False, drop_last=False,
    )
    pred_retain = []

    print('Calculating original model output for retain set')
    model = load_model(config).to(config['DEVICE']).eval()
    
    for retain_batch in tqdm(inference_retain):
        retain_batch = retain_batch.to(config['DEVICE'])
        pred_retain.append(model(retain_batch).detach().cpu())

    model.cpu()
    
    free_memory(model)

    target_retain = torch.concat(pred_retain, axis=0).numpy()

    target_forget = np.ones(
        (dataset['forget']['data'].shape[0], 10), 
        dtype=float
    )*coef_k #number for 1/10 --> equal probability

    if get_all:
        targets = np.concatenate(
            (target_retain, target_forget), 
            axis = 0
        )

        input_ = np.concatenate(
            (
                dataset['retain']['data'],
                dataset['forget']['data']
            ), axis=0
        )

        return input_, targets

    return get_unlearning_validation_train(
        data_retain=dataset['retain']['data'], target_retain=target_retain,
        data_forget=dataset['forget']['data'], target_forget=target_forget
    )


def get_unlearning_validation_train(
        data_retain: np.ndarray, target_retain: np.ndarray, 
        data_forget: np.ndarray, target_forget: np.ndarray,
        pct_split: float = 0.2
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:

    def shuffle_array(size: int) -> np.ndarray:
        indices = np.arange(size)
        np.random.shuffle(indices)
        
        return indices
    
    assert (pct_split > 0) & (pct_split < 1)
    
    #shuffle every array
    shuffled_index_retain = shuffle_array(data_retain.shape[0])
    shuffled_index_forget = shuffle_array(data_forget.shape[0])

    #length of validation
    num_val_retain = int(pct_split * len(shuffled_index_retain))
    num_val_forget = int(pct_split * len(shuffled_index_forget))

    #get data val/train
    val_data = np.concatenate(
        (
            data_retain[shuffled_index_retain[:num_val_retain]],
            data_forget[shuffled_index_forget[:num_val_forget]]
        ), axis=0
    )

    train_data = np.concatenate(
        (
            data_retain[shuffled_index_retain[num_val_retain:]],
            data_forget[shuffled_index_forget[num_val_forget:]]
        ), axis=0
    )

    #get target val/train
    val_target = np.concatenate(
        (
            target_retain[shuffled_index_retain[:num_val_retain]],
            target_forget[shuffled_index_forget[:num_val_forget]]
        ), axis=0
    )

    train_target = np.concatenate(
        (
            target_retain[shuffled_index_retain[num_val_retain:]],
            target_forget[shuffled_index_forget[num_val_forget:]]
        ), axis=0
    )

    return (train_data, train_target), (val_data, val_target)

def get_monitor_dataset(dataset: Dict[str, Dict[str, np.ndarray|list]]) -> Tuple[np.ndarray, np.ndarray]:
    
    output_dict = {
        "data": np.concatenate(
            [data_dict['data'] for _, data_dict in dataset.items()], axis=0
        ),
        "targets": np.concatenate(
            [data_dict['targets'] for _, data_dict in dataset.items()], axis = 0
        ),
        "labels_which_dataset": np.concatenate(
            [
                [name_dataset]*data_dict['data'].shape[0] 
                for name_dataset, data_dict in dataset.items()
            ], axis=0
        )
    }

    assert (
        (output_dict['data'].shape[0] == output_dict['targets'].shape[0]) & 
        (output_dict['targets'].shape[0] == output_dict['labels_which_dataset'].shape[0])
    )
 

    return output_dict
