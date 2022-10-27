import logging
from typing import Type, Any, Callable, Union, List, Optional
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torchvision import transforms as T

from dataset.mtbs import get_dataloaders

logger = logging.getLogger(__name__)

class PlDataset(LightningDataModule):

    def __init__(self, 
                # data_dir: str,
                # name: Optional[str] = None,
                # n_val: float = 0.1,
                # n_test: float = 0.2,
                # batch_size: int = 32,
                # num_workers: int = 4,
                # seed: int=0,
                cfg: dict,
                **kwargs):
        '''
        Lightning Data Module

        Parameter:
        ----------
        * name: optional, str
        * n_val: float, 0-1
        * n_test: float, 0-1
        * batch_size: int, default 32,
        * num_workers: int, default 4
        '''
        super().__init__()
        # self.name = name
        # self.n_val = n_val
        # self.n_test = n_test
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        self.dataloaders = get_dataloaders(cfg)
        

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        # return DataLoader(self.valset,
        #                   batch_size=self.batch_size,
        #                   num_workers=self.num_workers,
        #                   pin_memory=False,
        #                   drop_last=False)

        return self.dataloaders['valid']

    def test_dataloader(self):
        # return DataLoader(self.testset,
        #                   batch_size=self.batch_size,
        #                   num_workers=self.num_workers,
        #                   pin_memory=False,
        #                   drop_last=False)

        return self.dataloaders['test']

    
        



    
