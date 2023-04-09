from abc import abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from utils import *


class BASE(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
    
    @abstractmethod
    def forward(self, x):
        pass
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return get_errors(y, y_hat)

    def test_epoch_end(self, outputs):
        self.log_dict(stack_errors(outputs))

    def predict_step(self, batch, batch_idx) -> Any:
        x, y = batch
        y_hat = self(x)
        return y_hat
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
