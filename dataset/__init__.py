import logging
from pathlib import Path
from typing import Type, Any, Callable, Union, List, Optional
from xmlrpc.client import boolean
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torchvision import transforms as T

from dataset.mtbs import MTBS as Dataset

logger = logging.getLogger(__name__)


class PlDataset(LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.2,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 seed: int = 0,
                 use_aug: boolean=True,
                 train_mask: str='mtbs',
                 test_mask: str='mtbs',
                 **kwargs):
        '''
        Lightning Data Module

        Parameter:
        ----------
        * val_ratio: float, 0-1
        * test_ratio: float, 0-1
        * batch_size: int, default 32,
        * num_workers: int, default 4
        * use_aug: if use augmentation
        '''
        super().__init__()
        self.use_aug = use_aug
        self.batch_size = batch_size
        self.num_workers = num_workers
        train_files, val_files, test_files = Dataset.train_test_val_split(
            data_dir, test_ratio, val_ratio, seed)
        data_dir = Path(data_dir)

        if use_aug:
            logger.info('********* apply data augmentation ***********')
            transform_loc = T.Compose([
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=270),
                T.RandomResizedCrop(size=(256, 256), scale=(
                    0.2, 1), interpolation=T.InterpolationMode.NEAREST),
                # T.ToTensor()
            ])
            # TODO: non-location related augmentations
            self.trainset_aug = Dataset(train_files, data_dir, mask_src=train_mask, transform_loc=transform_loc, **kwargs)
        self.trainset = Dataset(train_files, data_dir, mask_src=train_mask, **kwargs)
        self.valset = Dataset(val_files, data_dir, mask_src=train_mask, **kwargs)
        self.testset = Dataset(test_files, data_dir, mask_src=test_mask, **kwargs)

    def train_dataloader(self):
        kwargs = dict(num_workers=self.num_workers, pin_memory=False)
        if self.use_aug:
            train_loader = CombinedLoader([
                DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, **kwargs),
                DataLoader(self.trainset_aug, batch_size=self.batch_size, shuffle=True, **kwargs),
            ])
        else:
            train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, **kwargs)

        return train_loader

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False,
                          drop_last=True)


    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False,
                          drop_last=False)

