from pathlib import Path
import logging
import wandb
import torch
import torch.nn as nn


class Revisit:
    def __init__(self) -> None:
        pass


api = wandb.Api()
logger = logging.getLogger(__name__)


def load_model_artifact(project: str, filters: dict) -> nn.Module:
    '''
    load torch.nn.Module from wandb artifact with given filters

    Parameters
    ----------
    * project: wandb project name
    * filters: dict, used to filter runs

    Return
    ----------
    * model: nn model
    '''
    runs = api.runs(project, filters=filters)
    print('len(runs)', len(runs))
    artifacts = runs[0].logged_artifacts()
    print(len(artifacts), artifacts)
    filtered = [atf for atf in artifacts if atf._sequence_name=='model_epoch199']
    model_path = filtered[0].download() #* use the first run
    model = torch.load(Path(model_path) / 'model.h5')
    return model

def inference(dataset, data_cfg,
                       seed: int,
                       loss_lambda: int,
                       device: torch.device):
    '''
    Inference on trained model. A pandas dataframe is returned, containing following columns:
    regression output, classification output (binarized), 
    regression output corrected by the classification output,
    regression label (log transformed), true count, regression count (transformed back), 
    regression*classification count (transformed back), input file name

    Parameters
    ------------
    * dataset: torch.Dataset object,
    * data_cfg: data config, attributes must have: batch_size, num_workers
    * loss_lambda: controls the contribution of the classification head in loss
    '''
    test_res = pd.DataFrame(columns=['regression_output', 'classification_output', 'regression_classfication_output', 'regression_label', 'file_name'])
    dataloader = DataLoader(dataset,
                    batch_size=data_cfg.batch_size,
                    num_workers=data_cfg.num_workers,
                    pin_memory=False,
                    drop_last=False)
    # *********************************** build model *********************************
    model = load_model_artifact('forest-coverage', 
                    filters={
                        'config.run_name': 'reg_only_test_loss_corrected' if loss_lambda==0 else 'reg_cf_test_loss_corrected',
                        'config.seed': seed,
                        'config.TRAIN_loss_lambda': loss_lambda,
                        'config.log_artifact_every': [180, 190, 195, 198, 199]
                    })
    logger.info("[Model] Loading model...")

    model = model.to(device=device)
    model.eval()
    for x, y, y_raw, path in dataloader:
        x = x.to(device)
        out_reg, out_cf = model(x)
        out_reg = out_reg.detach().squeeze()
        out_cf = torch.where(out_cf>0.5, 1, 0).squeeze()
        out_reg_cf = (out_reg * out_cf).detach().squeeze()
        reg_cf_count = (10**(out_reg_cf * 4)*out_cf).round().type(torch.int16) #torch.where(out_reg_cf>0, 10**(out_reg_cf*4), 0)#.type(torch.int16)
        reg_count = (10**(out_reg * 4)).round().squeeze().type(torch.int16)
        reg_count[reg_count==1] = 0
        df = pd.DataFrame({
            'regression_output': out_reg.detach().cpu(),
            'classification_output': out_cf.detach().cpu(),
            'regression_classfication_output': out_reg_cf.detach().cpu(),
            'regression_label': y.squeeze(),
            'true_count': y_raw,
            'reg_count': reg_count.cpu(),
            'reg_cf_count': reg_cf_count.cpu(),
            'file_name': path
        })
        test_res = pd.concat([test_res, df], ignore_index=True)
    return test_res