import logging
from typing import Any, Callable
from pytorch_lightning import LightningModule, LightningDataModule
import torch
import torch.nn.functional as F
from torchmetrics import R2Score
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss
from models import activation

from utils import acc_topk, accuracy
from trainer.ema import EMA

import torchmetrics as tm
from torchmetrics import JaccardIndex
from trainer.losses import get_criterion


logger = logging.getLogger(__name__)

class PLModel(LightningModule):
    def __init__(self, model: Any, dataset: LightningDataModule,
        n_epoch: int=100,
        lr: float=0.01,
        use_scheduler: bool=True,
        warmup: int=0,
        wdecay: float=0.01,
        batch_size: int=32,
        ema_used: bool=False, 
        ema_decay: float=0.9,
        loss_lambda: float=1,
        **kwargs
    ) -> None:
        '''
        Pytorch lightning trainer, Training process.

        Parameter
        ----------
        * model: NN model
        * dataset: pytorch-lightning LightningDataModule
        * n_epoch: number of training epochs
        * th: threshold on sigmoid for prediction
        * lr: learning rate
        * warmup: warmup epochs
        * wdecay: weight decay
        * batch_size: default 32
        * ema_used: if use ema
        * ema_decay: ema decay rate
        '''
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.n_epoch = n_epoch
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.wdecay = wdecay
        self.batch_size=batch_size
        self.warmup = warmup

        self.criterion = get_criterion(loss_type='CrossEntropyLoss', class_weights=[0.25,0.25,0.25,0.25])
        self.train_IoU = JaccardIndex(num_classes=4, average='none')
        self.val_IoU = JaccardIndex(num_classes=4)
        self.test_IoU = JaccardIndex(num_classes=4)

        # used EWA or not
        self.EMA = ema_used
        if self.EMA:
            self.EMA_DECAY = ema_decay,
            self.ema_model = EMA(self.model, ema_decay)
            logger.info("[EMA] initial ")
    

    def configure_optimizers(self):
        # optimizer
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.wdecay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        
        if self.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=10000, gamma=0.8),
                'interval': 'step',
                # 'strict': True,
            }
            return [optimizer], [scheduler]
        return optimizer

    def forward(self, x):
        return self.model.forward(x)

    def on_post_move_to_device(self):
        super().on_post_move_to_device()
        # used EWA or not
        # init ema model after model moved to device
        if self.EMA:
            self.ema_model = EMA(self.model, self.EMA_DECAY)
            logger.info("[EMA] initial ")

    def training_step(self, batch, batch_idx=None):
        if self.dataset.use_aug:
            batch0, batch1 = batch
            x = torch.cat([batch0[0], batch1[0]])
            y = torch.cat([batch0[1], batch1[1]])
        else:
            x, y = batch

        y = y.squeeze().long()
        out = self.model.forward(x) #NCHW
        loss = self.criterion(out[-1], y)

        y_pred = torch.argmax(self.model.activation(out[-1]), dim=1) # If use this, set IoU/F1 metrics activation=None

        mask = (y!=-1)
        y = y * mask
        class_iou = self.train_IoU(y_pred * mask, y)
        avg_iou = class_iou.mean()

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_IoU', avg_iou, on_step=False, on_epoch=True)
        for i, iou in enumerate(class_iou):
            self.log(f'train_IoU.{i}', iou, on_step=False, on_epoch=True)

        return loss

    def training_step_end(self, loss, *args, **kwargs):
        if self.EMA:
            self.ema_model.update_params()
        return loss

    def on_train_epoch_end(self) -> None:
        if self.EMA:
            self.ema_model.update_buffer()
            self.ema_model.apply_shadow()

    def validation_step(self, batch, batch_idx=None):
        x, y, = batch
        y = y.squeeze().long()
        out = self.model.forward(x) #NCHW
        loss = self.criterion(out[-1], y.squeeze().long())

        y_pred = torch.argmax(self.model.activation(out[-1]), dim=1) # If use this, set IoU/F1 metrics activation=None
        
        mask = (y!=-1)
        y = y * mask
        self.val_IoU(y_pred * mask, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_IoU', self.val_IoU, on_step=False, on_epoch=True)

        return y_pred.cpu(), y.cpu()

    def validation_epoch_end(self, *args, **kwargs):
        if self.EMA:
            self.ema_model.restore()

    def test_step(self, batch, batch_idx=None):
        x, y, = batch
        y = y.squeeze().long()
        out = self.model.forward(x) # NCHW
        loss = self.criterion(out[-1], y)

        y_pred = torch.argmax(self.model.activation(out[-1]), dim=1) # If use this, set IoU/F1 metrics activation=None
        
        mask = (y!=-1)
        y = y * mask
        self.test_IoU(y_pred * mask, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_IoU', self.test_IoU, on_step=False, on_epoch=True)

        return y_pred.cpu(), y.cpu()