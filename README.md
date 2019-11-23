# EAP-Net
Enhanced attention with pose assistance for Person-Reid

## Require
- [pytorch 1.0.0+](https://pytorch.org/)
- torchvision
- Python 3.6
- Numpy
- [yacs](https://github.com/rbgirshick/yacs)
- [ignite](https://pypi.org/project/pytorch-ignite/)

## DataSet
- Market-1501: [[google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op)
- DukeMTMC-reid
- [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
- MSMT17

## Test

1. Prepare pretrained model.
  In this project, We only provide the checkpoints and test code for our model.The code for model training will be updated later.If you want to test our model, you need to download the model's checkpoints in advance.

2. Change the config file.
  Our model test parameters is saved at  `./configs/inference.yml`. You need to update your own test parameters to this file, for example`path to the checkpoint`, `dataset name`.

3. Test command
  You can test our model's performance by running this command.
  `python eval/test.py`

## Experiment Results
1. EAP-net base on Res101

|DataSet | Rank@1 | Rank@5 | Rank@10 | mAP| Checkpoint|
| -------- | ----- | ----- | ----- | ---- | ---- |
| [Market-1501] | 96.3% | 0% | 0% | 91.1% |  [res101-market] |
| [DukeMTMC-reid] | 91.0% | 0% | 0% | 83.0% | [res101-duke] |
| [CUHK03-Lableled] | 89.8% | 0% | 0% | 85.8% | [res101-label] |
| [CUHK03-Detected] | 87.0% | 0% | 0% | 82.5% | [res101-detect] |
| [MSMT17] | 86.4% | 0% | 0% | 65.9% | [res101-msmt] |
