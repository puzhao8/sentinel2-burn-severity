from typing import Union, Any, List
from omegaconf.listconfig import ListConfig
import wandb
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics import ConfusionMatrix
import plotly.express as px

from utils import fig2html

blues = px.colors.sequential.Blues


class CfMatrixLogger(Callback):

    def __init__(self, log_every: Union[int, list[int]], data_module, data_name: str='val') -> None:
        '''
        A pytorch lightning Callback for logging confusion matrix of specified dataset. 
        It accumulates confusion matrix for specified epochs.
        !! drop last must be False

        Parameter
        ----------
        * log_every: int or list of int
        * data_module: pl data module, 
            * data_module.classes: list of names of each class, used for the confusion matrix labels
            * data_module.batch_size: used with data_size to determine if current batch is the last batch
        * data_name: val or test, choose which dataset to plot confusion on
        '''
        super().__init__()
        self.log_every = log_every
        assert isinstance(log_every, (int, list, ListConfig, np.ndarray)
                          ), 'invalid type of log_every, must be list or int'

        self.batch_size = data_module.batch_size
        assert not data_module.drop_last, 'drop_last mush be False!'

        self.class_names = data_module.classes
        self.n_class = len(self.class_names)
        self.cfmatrix = torch.zeros((self.n_class, self.n_class))
        self.comput_cfmat = ConfusionMatrix(num_classes=self.n_class)
    
        self.data_name = data_name
        if data_name == 'val':
            self.data_size = len(data_module.valset)
            self.on_validation_batch_end = self.callback
        elif data_name == 'test':
            self.data_size = len(data_module.testset)
            self.on_test_batch_end = self.callback
        else:
            raise Exception('data_name must be val or test!')    
        

    def callback(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        current_epoch = trainer.current_epoch
        if isinstance(self.log_every, int):
            iflog = (current_epoch + 1) % self.log_every == 0
        else:
            iflog = current_epoch in self.log_every
        
        if iflog:
            v = self.comput_cfmat(*outputs)
            self.cfmatrix += v
            if self.batch_size * (batch_idx + 1) >= self.data_size:  # the last batch, plot the confusion matrix and reset self.cfmatrix
                fig = px.imshow(self.cfmatrix,
                                x=self.class_names, y=self.class_names,
                                aspect="equal",
                                text_auto=".2f",
                                # hoverinfo='z',
                                # annotation_text=np.around(cfmatrix, 2),
                                color_continuous_scale=[
                                    [0, blues[0]],
                                    [1./1000, blues[2]],
                                    [1./100, blues[4]],
                                    [1./10, blues[6]],
                                    [1., blues[8]],
                                ])

                fig.update_layout(
                    width=1000,
                    height=900,
                    title={
                        'text': f'Confusion Matrix (epoch {current_epoch})',
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    xaxis_title="Predicted label",
                    yaxis_title="True label",
                    legend_title="Legend Title",
                )
                # fig.show()
                html = fig2html(fig)
                wandb.log({f'{self.data_name}.cfmatrix epoch{current_epoch}': wandb.Html(html)})
                self.cfmatrix = torch.zeros((self.n_class, self.n_class))

