import os

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from src.utils import get_all_config
from src.model.unlearn import UnLearner
from src.dataset.dataset import (
    get_unlearning_dataset, get_monitor_dataset, 
    ImageDataset, MonitorImageDataset
)
from src.dataset.utility import load_dataset

def run_experiment() -> None:

    all_config = get_all_config()

    config, config_model = all_config['config'], all_config['config_model']

    dataset = load_dataset(config)
    monitor_data = get_monitor_dataset(dataset)
    monitor_dataset = MonitorImageDataset(monitor_data)

    def monitor_dataloader():
        return DataLoader(
            monitor_dataset, 
            batch_size=128, shuffle=False, num_workers=1,
            pin_memory=False, drop_last=False,
        )

    (train_data, train_target), (val_data, val_target) = get_unlearning_dataset(config, dataset)

    unlearning_train_dataset = ImageDataset(train_data, train_target)
    unlearning_valid_dataset = ImageDataset(val_data, val_target)

    unlearning_train = DataLoader(
        unlearning_train_dataset, 
        batch_size=128, shuffle=True, num_workers=3,
        pin_memory=True, drop_last=True,
    )
    unlearning_valid = DataLoader(
        unlearning_valid_dataset, 
        batch_size=128, shuffle=False, num_workers=1,
        pin_memory=False, drop_last=False,
    )

    model = UnLearner(config=config, config_model=config_model, monitor_dataset=monitor_dataloader)

    unlearn_trainer = pl.Trainer(
        max_epochs=config_model['max_epochs'],
        max_steps=-1,
        accelerator=config['DEVICE'],
        enable_progress_bar=False,
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        logger=False
    )
    unlearn_trainer.fit(model, unlearning_train, unlearning_valid)

    unlearn_trainer.save_checkpoint(os.path.join(config['MODEL_FOLDER'], "model.ckpt"))