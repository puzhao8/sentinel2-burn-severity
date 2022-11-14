import logging
from pathlib import Path
import torch
import wandb
import hydra

from sklearn.preprocessing import LabelEncoder



api = wandb.Api()
logger = logging.getLogger(__name__)
label_encoder = LabelEncoder()


class Summary:
    '''
    Use wandb.api to revisit trained models or wandb run summaries.
    '''

    def __init__(self, data_path, wandb_project: str, run_name: str, seed: int = 0, use_gpu: bool = True) -> None:
        '''
        Parameters
        ---------------
        * wandb_project: wandb project name, used to init wandb run
        * run_name: current run name, used to init wandb run
        * seed: random seed
        * use_gpu: bool, used to init device
        '''
        self.root = Path(hydra.utils.get_original_cwd())
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.seed = seed
        self.wandb_project = wandb_project
        self.run_name = run_name
        self.path = data_path

    






