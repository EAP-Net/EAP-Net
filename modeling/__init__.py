# encoding: utf-8

from .eap import Baseline
from .LR import Baseline as Draft_Baseline


def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'resnet50':
        model = Baseline(num_classes, cfg.DATASETS.NAMES, cfg.MODEL.LAST_STRIDE, back_num=0)
    else:
        model = Draft_Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.DATASETS.NAMES)
    return model

