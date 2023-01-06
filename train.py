import logging
import os
import sys
import torch
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from dataset import PlDataset

# from models.resnet18 import ResNet
from models import get_model

from trainer._base_trainer import PLModel
from trainer.log_cfmatrix import CfMatrixLogger
from trainer.log_artifacts import ArtifactLogger
from utils import flat_omegadict, set_random_seed


def isdebugging():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        print('No sys.gettrace')
    elif gettrace():
        print('Hmm, Big Debugger is watching me')
        return True
    return False


@hydra.main(config_path='./config', config_name='mtbs')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))

    if (device_n := CFG.CUDA_VISIBLE_DEVICES) is not None:
        # import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_n)

    if isdebugging():
        os.environ['CUDA_VISIBLE_DEVICES'] = '9'
        print('CUDA_VISIBLE_DEVICES: ', os.environ['CUDA_VISIBLE_DEVICES'])

    os.environ['HYDRA_FULL_ERROR'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available()
                          and CFG.use_gpu else 'cpu')

    # # For reproducibility, set random seed
    set_random_seed(CFG.seed)

    # get datasets
    # data_dir = Path(hydra.utils.get_original_cwd()) / CFG.DATASET.data_dir
    # CFG.DATASET.data_dir = str(data_dir)

    data_module = PlDataset(seed=0, **CFG.DATA)  # * fix dataset

    # build model
    model = get_model(CFG)
    # logger.info("[Model] Building model -- input dim: {}, hidden nodes: {}, out dim: {}"
    #                             .format(input_dim, CFG.MODEL.h_nodes, CFG.MODEL.out_dim))

    litModel = PLModel(model, data_module,
                       batch_size=CFG.DATASET.batch_size, **CFG.TRAIN)

    wandb_logger = WandbLogger(
        project=CFG.wandb_project,
        name=CFG.run_name,
        config=flat_omegadict(CFG),
        job_type='train'
    )

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val_IoU',
        dirpath='checkpoints/',
        filename='mtbs-{epoch:02d}-{val_IoU:.2f}',
        save_top_k=CFG.save_model_top_k,
        every_n_epochs=CFG.save_model_every,
        save_last=CFG.save_model_last,
        mode='max',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor, checkpoint_callback]

    if CFG.save_model_top_k or CFG.save_model_every or CFG.save_model_last:
        model_artifact_logger = ArtifactLogger()
        callbacks.append(model_artifact_logger)
        
    if (log_every := CFG.log_cfmatrix_every) is not None:
        artifact_logger = CfMatrixLogger(log_every, data_module=data_module, data_name='val')
        callbacks.append(artifact_logger)

    trainer = Trainer(
        # accelerator="ddp",  # if torch.cuda.is_available() else 'ddp_cpu',
        callbacks=callbacks,
        logger=wandb_logger,
        # checkpoint_callback=False if CFG.debug else checkpoint_callback,
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=CFG.TRAIN.n_epoch,
        # gradient_clip_val=1,
        enable_progress_bar=False
    )

    logger.info("======= Training =======")
    print('cuda available', torch.cuda.is_available())
    trainer.fit(litModel, datamodule=data_module)

    logger.info("======= Testing =======")
    trainer.test(litModel, datamodule=data_module)

    wandb.finish()


if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
