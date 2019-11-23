# encoding: utf-8

import torch
from ignite.engine import Engine
from utils.reid_metric import EvalMetric


def create_supervised_evaluator(model, metrics,
                                device=None, num_gpus=1):
    if num_gpus > 1:
        device_id = []
        for i in range(num_gpus):
            device_id.append(i)
        model = torch.nn.DataParallel(model, device_ids=device_id)

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.cuda()
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query,
        num_gpus
):
    device = cfg.MODEL.DEVICE

    print("EAP-Net.inference")
    print("Start inferencing")
    evaluator = create_supervised_evaluator(model, metrics={'eval_metric': EvalMetric(num_query, cfg.DATASETS.NAMES)},
                                            device=device, num_gpus=num_gpus)

    evaluator.run(val_loader)
    cmc = evaluator.state.metrics['eval_metric']
    # print(cmc)

