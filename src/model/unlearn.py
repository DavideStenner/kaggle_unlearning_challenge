import torch

import pytorch_lightning as pl

from torch import nn
from typing import Dict, Tuple

from src.dataset.dataset import load_model

class UnLearner(pl.LightningModule):
    def __init__(self, config:dict, config_model: dict):
        super().__init__()

        self.config = config
        self.config_model = config_model
        self.lr = config_model['lr']

        self.save_hyperparameters()

        self.model = load_model(config)
        self.criterion = nn.MSELoss()
        
        self.step_outputs = {
            'train': [],
            'val': [],
            'test': []
        }

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

    def __share_eval_res(self, mode: str):
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
            pass

        if self.trainer.sanity_checking:
            pass
        else:
            print(', '.join(metric_message_list))
            self.log_dict(res_dict)
            
        #free memory
        self.step_outputs[mode].clear()

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
