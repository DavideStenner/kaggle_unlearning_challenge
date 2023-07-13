import torch

import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, Optional, Callable

from src.dataset.dataset import load_model

class UnLearner(pl.LightningModule):
    def __init__(self, config:dict, config_model: dict, monitor_dataset: Dataset):
        super().__init__()

        self.config = config
        self.config_model = config_model
        self.lr = config_model['lr']

        self.save_hyperparameters()

        self.monitor_dataset = monitor_dataset
        self.model = load_model(config)
        self.criterion = nn.MSELoss()
        
        self.step_outputs = {
            'train': [],
            'val': [],
            'test': []
        }

    def __monitor_dataloader(self):
        return self.monitor_dataset()
    
    def __loss_step(self,
        pred: torch.tensor | Dict, 
        labels: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        loss = self.criterion(pred, labels)
        return loss, pred, labels
    
    def training_step(self, batch, batch_idx):

        input_, labels = batch

        pred = self.forward(input_)

        loss, _, _ = self.__loss_step(pred, labels)
        self.step_outputs['train'].append(
            {'loss': loss}
        )

        return loss
    
    def validation_step(self, batch, batch_idx):

        input_, labels = batch
        pred = self.forward(input_)

        loss, pred, labels = self.__loss_step(pred, labels)
        self.step_outputs['val'].append(
            {'loss': loss, 'pred': pred, 'labels': labels}
        )

    def on_training_epoch_end(self):
        self.__share_eval_res('train')
        
    def on_validation_epoch_end(self):
        self.__share_eval_res('val')

    def on_test_epoch_end(self):
        self.__share_eval_res('test')

    def __inspect_unlearning(self) -> Dict[str, float]:

        unlearning_pred, unlearning_target, unlearning_dataset = [], [], []

        for batch in self.__monitor_dataloader():
            img_input, target, label_dataset = batch

            img_input = img_input.to(self.config['DEVICE'])
            
            unlearning_pred.append(self.forward(img_input))
            unlearning_target.append(target)
            unlearning_dataset.append(label_dataset)
        
        unlearning_pred = torch.cat(unlearning_pred)
        unlearning_target = torch.cat(unlearning_target).numpy()
        unlearning_dataset = np.concatenate(unlearning_dataset, axis=0)

        unlearning_pred = (
            unlearning_pred.numpy() if self.config['DEVICE'] == 'cpu'
            else unlearning_pred.cpu().numpy()
        )
        results = {}

        for dataset_label in np.unique(unlearning_dataset):
            mask_dataset = unlearning_dataset == dataset_label

            pred_label = np.argmax(unlearning_pred[mask_dataset], axis=1)
            accuracy = (pred_label==unlearning_target[mask_dataset]).mean()
            results[dataset_label] = accuracy

        results['diff_forget'] = np.abs(results['forget'] - results['test'])
        results['diff_retain'] = np.abs(results['retain'] - results['train'])


        return results

    def __share_eval_res(self, mode: str):
        if self.trainer.sanity_checking:
            pass
        else:

            outputs = self.step_outputs[mode]
            loss = [out['loss'].reshape(1) for out in outputs]
            loss = torch.mean(torch.cat(loss))
            
            #initialize performance output
            res_dict = {
                f'{mode}_loss': loss
            }
            metric_message_list = [
                f'step: {self.trainer.global_step}',
                f'{mode}_loss: {loss:.5f}'
            ]
        
            #evaluate on all dataset
            if mode != 'train':
                metric_score = {}

                preds = [out['pred'] for out in outputs]
                preds = torch.sigmoid(torch.cat(preds))
                
                labels = [out['labels'] for out in outputs]
                labels = torch.cat(labels)
            
                # metric_score = self.__classifier_metric_step(preds, labels)
                
                metric_score.update(self.__inspect_unlearning())
                
                #calculate every metric on all batch
                metric_message_list += [
                    f'{mode}_{metric}: {metric_value:.5f}'
                    for metric, metric_value in metric_score.items()
                ]
                #get results
                res_dict.update(
                    {
                        f'{mode}_{metric}': metric_value
                        for metric, metric_value in metric_score.items()
                    }
                )

            print(', '.join(metric_message_list))
            self.log_dict(res_dict)
            
        #free memory
        self.step_outputs[mode].clear()

    def __classifier_metric_step(self, pred: torch.tensor, labels: torch.tensor) -> dict:

        labels = (
            labels.numpy() if self.config['DEVICE'] == 'cpu'
            else labels.cpu().numpy()
        )
        pred = (
            pred.numpy() if self.config['DEVICE'] == 'cpu'
            else pred.cpu().numpy()
        )
        pred_idx = np.argmax(pred, axis=1)
        label_idx = np.argmax(labels, axis=1)

        accuracy = (pred_idx==label_idx).mean()

        return {'accuracy': accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config_model['max_epochs'])
        optimizer_dict = dict(
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

        return optimizer_dict

    def forward(self, inputs: torch.tensor):
        output = self.model(inputs)
                
        return output

    def predict_step(self, batch: torch.tensor, batch_idx: int):
        pred = self.forward(batch)

        return pred
