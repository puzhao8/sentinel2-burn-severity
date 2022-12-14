
from models.resnet import ResNet
from models.unet import UNet, UNet_dualHeads

model_zoo = {
    'UNet': UNet,
    'UNet_dualHeads': UNet_dualHeads
}


def get_model(cfg):

    ########################### COMPUTE INPUT & OUTPUT CHANNELS ############################
    print("Satellites: ", cfg.DATA.satellites)
    print("num_classes:", cfg.MODEL.NUM_CLASS)

    # cfg.MODEL.NUM_CLASS = cfg.MODEL.cfg.MODEL.NUM_CLASS

    INPUT_CHANNELS_DICT = {}
    INPUT_CHANNELS_LIST = []
    for sat in cfg.DATA.satellites:
        INPUT_CHANNELS_DICT[sat] = len(list(cfg.DATA.input_bands[sat]))
        if cfg.DATA.stacking: INPUT_CHANNELS_DICT[sat] = len(cfg.DATA.prepost) * INPUT_CHANNELS_DICT[sat]
        INPUT_CHANNELS_LIST.append(INPUT_CHANNELS_DICT[sat])
    
    ########################### MODEL SELECTION ############################
    if cfg.MODEL.ARCH in model_zoo.keys():
        INPUT_CHANNELS = sum(INPUT_CHANNELS_LIST)
        MODEL = model_zoo[cfg.MODEL.ARCH]
        return MODEL(INPUT_CHANNELS, 
                    num_classes=cfg.MODEL.NUM_CLASS, 
                    topo=cfg.MODEL.TOPO,
                    use_deconv=cfg.MODEL.USE_DECONV) #'FC-EF'

    