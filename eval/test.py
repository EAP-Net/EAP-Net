# encoding: utf-8

import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from configs import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model


def main():

    num_gpus = int(os.environ["GPU_NUM"]) if "GPU_NUM" in os.environ else 1

    cfg.merge_from_file('configs/inference.yml')
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR

    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))

    inference(cfg, model, val_loader, num_query, num_gpus)


if __name__ == '__main__':
    main()
