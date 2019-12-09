# EAP-Net
Enhanced Attention with Pose assistance for Person-reID.

The figure dipicts the retrieval results bewteen EAP-Net(the lower row) and global learning baseline(the upper row), more examples and EAP-Net's attention figures are exhibited in [`More Visual Results.md`](https://github.com/EAP-Net/EAP-Net/blob/master/More%20Visual%20Results.md).
<p align="center"><img src="https://github.com/EAP-Net/EAP-Net/blob/master/query_result.png" /></p>

## Require
- [pytorch 1.0.0+](https://pytorch.org/)
- torchvision
- Python 3.6
- Numpy
- [yacs](https://github.com/rbgirshick/yacs)
- [ignite](https://pypi.org/project/pytorch-ignite/)

## DataSet
- Market-1501
- DukeMTMC-reid
- [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
- MSMT17

## Test

1. Prepare pretrained model.  
  
    In this project, We provide the checkpoints and test code for our model. The code for model training will be updated later.If you want to evaluate our model, you can download the checkpoints and run the test code on your machine.

2. Change the config file.  
  
    Our model test parameters is saved at  [`./configs/inference.yml`](https://github.com/EAP-Net/EAP-Net/blob/master/configs/inference.yml). You can update your own test parameters to this file, for example`path to the checkpoint`, `dataset name`.

3. Test command.  
  
    You can test our model's performance by running this command.  
    ```bash
    python eval/test.py
    ```

4. DataSet prepared question.  
  
    We use [`./data/dataset/cuhk03.py`](https://github.com/EAP-Net/EAP-Net/blob/master/data/datasets/cuhk03.py) to divide the cuhk03 dataset into the labeled and detected. If you want to test our model, you should download the `.mat` file in this [project](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03).

## Experiment Results
1. EAP-net base on Res101

|DataSet | Rank@1 | Rank@5 | Rank@10 | mAP| Checkpoint|Last updated|
| -------- | ----- | ----- | ----- | ---- | ---- |---- |
| Market-1501 | 96.3% | 98.8% | 99.2% | 91.1% |  [[google]](https://drive.google.com/file/d/1KdOO0Onp20tJhRgtGBvHF6B60iqAfBzh/view?usp=sharing) |23/11/2019 |
| DukeMTMC-reid | 91.0% | 96.0% | 97.4% | 83.0% | [[google]](https://drive.google.com/file/d/1Qc-QTtj_1c8dyZ6jUK0JWIVZ9U9VcXh9/view?usp=sharing) |23/11/2019 |
| CUHK03-Lableled | 89.8% | 97.7% | 98.6% | 85.8% | [[google]](https://drive.google.com/file/d/1FY3FKA8E-GWwrzJdFb-7Pqv0ZRnRXuGS/view?usp=sharing) |23/11/2019 |
| CUHK03-Detected | 87.0% | 96.8% | 98.3% | 82.5% | [[google]](https://drive.google.com/file/d/14cc1FQs4aYbb3e16j3Q2I-kHh_yjg7pY/view?usp=sharing) |23/11/2019 |
| MSMT17 | 86.4% | 92.6% | 94.5% | 65.9% | [[google]](https://drive.google.com/file/d/192JxOptm8wz2OJxjxmUK8DXFMLQ5MIE7/view?usp=sharing) |23/11/2019 |

2. EAP-net base on Res50

|DataSet | Rank@1 | Rank@5 | Rank@10 | mAP| Checkpoint| Last updated|
| -------- | ----- | ----- | ---- | ---- | ---- | ---- |
| Market-1501 | 95.8% | 98.7% |99.1% | 90.1% |  [[google]](https://drive.google.com/file/d/1Cs1kFayanwIQtJGR5xEgbLipXFRF_NWO/view?usp=sharing) | 09/12/2019 |
| DukeMTMC-reid | 90.8% | 95.8 % | 96.9 % | 82.0% | [[google]](https://drive.google.com/file/d/16QZBlRv2YOC5-u1U675pVx6GwGKQscAp/view?usp=sharing) | 29/11/2019 |
| CUHK03-Lableled | 87.9% | 97.8% | 98.6% |83.4% | [[google]](https://drive.google.com/file/d/1ZX4XhrDqLyWen8yykzeZmEcMkgLh9a9A/view?usp=sharing) | 01/12/2019 |
| CUHK03-Detected | 85.2% | 96.5% | 98.0% |80.4% | [[google]](https://drive.google.com/file/d/1Vf97rFyi4zCk3xs8HWZ5LfzH-Jiub1PZ/view?usp=sharing) | 01/12/2019 |

