# EAP-Net
Enhanced Attention with Pose assistance for Person-reID

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

|DataSet | Rank@1 | Rank@5 | Rank@10 | mAP| Checkpoint|
| -------- | ----- | ----- | ----- | ---- | ---- |
| Market-1501 | 96.3% | 98.8% | 99.2% | 91.1% |  [[google]](https://drive.google.com/file/d/1KdOO0Onp20tJhRgtGBvHF6B60iqAfBzh/view?usp=sharing) |
| DukeMTMC-reid | 91.0% | 96.0% | 97.4% | 83.0% | [[google]](https://drive.google.com/file/d/1Qc-QTtj_1c8dyZ6jUK0JWIVZ9U9VcXh9/view?usp=sharing) |
| CUHK03-Lableled | 89.8% | 97.7% | 98.6% | 85.8% | [[google]](https://drive.google.com/file/d/1FY3FKA8E-GWwrzJdFb-7Pqv0ZRnRXuGS/view?usp=sharing) |
| CUHK03-Detected | 87.0% | 96.8% | 98.3% | 82.5% | [[google]](https://drive.google.com/file/d/14cc1FQs4aYbb3e16j3Q2I-kHh_yjg7pY/view?usp=sharing) |
| MSMT17 | 86.4% | 92.6% | 94.5% | 65.9% | [[google]](https://drive.google.com/file/d/192JxOptm8wz2OJxjxmUK8DXFMLQ5MIE7/view?usp=sharing) |

2. EAP-net base on Res50

|DataSet | Rank@1 | Rank@5 | mAP| Checkpoint| Last updated|
| -------- | ----- | ----- | ---- | ---- | ---- |
| Market-1501 | 95.8% | 98.6% | 89.9% |  [[google]](https://drive.google.com/file/d/1UYHc2ZIB4oin4IxlHNOcl1_4z8K1GcnN/view?usp=sharing) | 26/11/2019 |
| DukeMTMC-reid | 90.5% | 95.6 % | 81.7% | [[google]](https://drive.google.com/file/d/1MPB4sj8zW5MAIgSM07nZ_xpO_nupVunG/view?usp=sharing) | 26/11/2019 |
| CUHK03-Lableled | 87.4% | 97.7% |82.8% | [[google]](https://drive.google.com/file/d/1mdb5KGrT1zvcC17t6o4gxfWMEnJgiPh5/view?usp=sharing) | 23/11/2019 |
| CUHK03-Detected | 84.4% | 96.1% |80.0% | [[google]](https://drive.google.com/file/d/1wThu7RdCDlxJW6mN-RNJZzlTivDvdZck/view?usp=sharing) | 23/11/2019 |

## EAP-Net's Attention
The figure depicts the attention heat maps of global learning branch (GL), pose assistant branch (PA), enhanced pose assistant branch (EPA), PA+EPA and EAP-Net. Compared with PA, EPA enhances the attention on local points by feature drop. By integrating PA and EPA into GL, EAP-Net reduces the attention on clothes, special items or background.
<img width="50%" src="https://github.com/EAP-Net/EAP-Net/blob/master/heatmap.png" />

## Query result bewteen EAP-Net with global learning baseline
<img width="50%" src="https://github.com/EAP-Net/EAP-Net/blob/master/query_results.png" />
