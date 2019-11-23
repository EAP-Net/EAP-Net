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
  ```bash
  python eval/test.py
  ```

4. DataSet prepared question
  We use `./data/dataset/cuhk03.py` to divide the cuhk03 dataset into the labeled and detected. If you want to test our model, you should download the `.mat` file in this [project](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03).

## Experiment Results
1. EAP-net base on Res101

|DataSet | Rank@1 | Rank@5 | Rank@10 | mAP| Checkpoint|
| -------- | ----- | ----- | ----- | ---- | ---- |
| [Market-1501] | 96.3% | 98.8% | 99.2% | 91.1% |  [[google]](https://drive.google.com/file/d/1KdOO0Onp20tJhRgtGBvHF6B60iqAfBzh/view?usp=sharing) |
| [DukeMTMC-reid] | 91.0% | 96.0% | 0% | 83.0% | [[google]](https://drive.google.com/file/d/1Qc-QTtj_1c8dyZ6jUK0JWIVZ9U9VcXh9/view?usp=sharing) |
| [CUHK03-Lableled] | 89.8% | 97.7% | 0% | 85.8% | [[google]](https://drive.google.com/file/d/1FY3FKA8E-GWwrzJdFb-7Pqv0ZRnRXuGS/view?usp=sharing) |
| [CUHK03-Detected] | 87.0% | 96.8% | 0% | 82.5% | [[google]](https://drive.google.com/file/d/14cc1FQs4aYbb3e16j3Q2I-kHh_yjg7pY/view?usp=sharing) |
| [MSMT17] | 86.4% | 0% | 0% | 65.9% | [[google]](https://drive.google.com/file/d/192JxOptm8wz2OJxjxmUK8DXFMLQ5MIE7/view?usp=sharing) |

2. EAP-net base on Res50

|DataSet | Rank@1 | Rank@5 | Rank@10 | mAP| Checkpoint|
| -------- | ----- | ----- | ----- | ---- | ---- |
| [Market-1501] | 95.8% | 98.6% | 99.1% | 90.1% |  [Uploading soon] |
| [DukeMTMC-reid] | 90.8% | 0% | 0% | 81.7% | [Uploading soon] |
| [CUHK03-Lableled] | 87.4% | 0% | 0% | 82.8% | [res50-label] |
| [CUHK03-Detected] | 83.7% | 0% | 0% | 79.8% | [Uploading soon] |
