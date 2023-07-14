import os
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import DataLoader

from src.utils import get_all_config
from src.loss.loss import compute_losses, create_mia_dataset, simple_mia
from src.plot.plot import train_test_loss_plot, attack_score_plot
from src.dataset.utility import load_weights, load_all_model, load_dataset
from src.model.unlearn import UnLearner
from src.model.utils import get_weights_diff
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

    torch.save(
        unlearn_trainer.model.model.state_dict(), 
        os.path.join(config['MODEL_FOLDER'], "model_weights.pth")
    )


def eda() -> None:
    
    all_config = get_all_config()
    config, config_model = all_config['config'], all_config['config_model']

    original_weight, unlearn_weight = load_weights()

    model_dict = load_all_model(original_weight=original_weight, unlearn_weight=unlearn_weight)
    original_model, unlearn_model = model_dict['original'], model_dict['unlearn']

    diff_res = get_weights_diff(original_weight, unlearn_weight)

    fig = sns.catplot(diff_res, x='layer', y='diff_num', kind='bar')
    fig = fig.set_xticklabels(rotation=90)
    fig.savefig(
        os.path.join(config['PLOT_FOLDER'], 'weights_diff.png')
    )

    dataset = load_dataset(config)

    train_loader = DataLoader(
        ImageDataset(**dataset['train']), 
        batch_size=128, shuffle=False, num_workers=3,
        pin_memory=False, drop_last=False,
    )
    test_loader = DataLoader(
        ImageDataset(**dataset['test']), 
        batch_size=128, shuffle=False, num_workers=3,
        pin_memory=False, drop_last=False,
    )
    forget_loader = DataLoader(
        ImageDataset(**dataset['forget']), 
        batch_size=128, shuffle=False, num_workers=3,
        pin_memory=False, drop_last=False,
    )
    
    print('\n\nCalculating train/test losses')
    train_losses_original = compute_losses(
        model=original_model, loader=train_loader, config=config
    )
    test_losses_original = compute_losses(
        model=original_model, loader=test_loader, config=config
    )

    fig = train_test_loss_plot(train_losses=train_losses_original, test_losses=test_losses_original)
    fig.savefig(
        os.path.join(config['PLOT_FOLDER'], 'train_test_loss.png')
    )

    print('\nCalculating forget losses')
    forget_losses_original = compute_losses(
        model=original_model, loader=forget_loader, config=config
    )
    samples_mia, labels_mia = create_mia_dataset(
        forget_losses=forget_losses_original, test_losses=test_losses_original
    )
    mia_scores_original = simple_mia(
        sample_loss=samples_mia, members=labels_mia, random_state=config['RANDOM_STATE']
    )

    print(
        f"\nThe MIA attack on original model has an accuracy of {mia_scores_original.mean():.3f} on forgotten vs unseen images"
    )


    forget_losses_unlearn = compute_losses(
        model=unlearn_model, loader=forget_loader, config=config
    )
    test_losses_unlearn = compute_losses(
        model=unlearn_model, loader=test_loader, config=config
    )
    samples_mia, labels_mia = create_mia_dataset(
        forget_losses=forget_losses_unlearn, test_losses=test_losses_unlearn
    )
    mia_scores_unlearn = simple_mia(
        sample_loss=samples_mia, members=labels_mia, random_state=config['RANDOM_STATE']
    )
 
    print(
        f"\nThe MIA attack on unlearned model has an accuracy of {mia_scores_unlearn.mean():.3f} on forgotten vs unseen images"
    )

    fig = attack_score_plot(
        config=config,
        original_model=original_model, unlearn_model=unlearn_model,
        test_loader=test_loader, forget_loader=forget_loader,
        mia_scores=mia_scores_original, mia_scores_ft=mia_scores_unlearn
    )

    fig.savefig(
        os.path.join(config['PLOT_FOLDER'], 'final_res.png')
    )
